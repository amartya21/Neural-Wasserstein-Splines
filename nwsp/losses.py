"""Fit term of NWSP: debiased Sinkhorn divergence, multi-bandwidth Gaussian MMD and their automatic settings.

For the retained snapshots mu_1, ..., mu_K at normalised times t_1, ..., t_K the generator's marginals rho_t are
fitted with

    L_fit = (1/K) sum_k [ S_eps(rho_{t_k}, mu_k) + mmd_weight MMD^2(rho_{t_k}, mu_k) ]

on minibatches (b_fit generated and b_fit observed points per snapshot and step).

* S_eps is the debiased Sinkhorn divergence on the squared Euclidean cost c(x, y) = |x - y|^2,

      S_eps(alpha, beta) = OT_eps(alpha, beta) - 1/2 OT_eps(alpha, alpha) - 1/2 OT_eps(beta, beta).

  The dual potentials are computed by n_iter log-domain iterations without gradient and refined by one
  differentiable iteration on the cost matrices (no backpropagation through the iterations). The refinement is
  differentiated through both potentials of OT_eps(alpha, beta) and both arguments of C(x, x), so at converged
  potentials its gradient with respect to the generated points is that of 2 S_eps, while the returned value is S_eps.
* MMD^2 is the plug-in estimate with the Gaussian kernels k_s(x, y) = exp(-|x - y|^2 / (2 s^2)), averaged over
  the bandwidths s = (0.25, 0.5, 1, 2, 4) x the median pairwise distance of the retained data.
* eps = max(1e-3, 0.02 mean_k sum_j Var(frame_k[:, j])) and b_fit = min(128, smallest snapshot) by default.

Every function returns tensors on the device of its inputs and never synchronises with the host. On CUDA the potential
iterations are captured once into a CUDA graph and replayed: the same kernels on the same inputs, so the potentials
equal those of the eager loop bit for bit, without the kernel-launch cost.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

MMD_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
"""Bandwidths of the MMD: multiples of the median pairwise distance of the retained data."""

_LOG_FLOOR = 1e-30  # floor of the weights inside log(a), log(b)


# ------------------------------------------------------------------------------------------------- helpers
def _sqdist(x: Tensor, y: Tensor) -> Tensor:
    """Squared Euclidean costs ``C_ij = |x_i - y_j|^2`` as ``cdist(x, y) ** 2``; (n, d), (m, d) -> (n, m)."""
    return torch.cdist(x, y, p=2) ** 2


def _uniform(n: int, like: Tensor) -> Tensor:
    """Uniform probability vector of length n with the dtype and device of ``like``."""
    return torch.full((n,), 1.0 / n, dtype=like.dtype, device=like.device)


def _log_weights(w: Tensor) -> Tensor:
    return torch.log(w.clamp_min(_LOG_FLOOR))


def _cross_potentials(C: Tensor, la: Tensor, lb: Tensor, eps: float, n_iter: int) -> tuple[Tensor, Tensor]:
    """Log-domain Sinkhorn iterations for OT_eps(a, b) on the cost C (..., n, m), started at 0:

        f_i <- -eps log sum_j b_j exp((g_j - C_ij) / eps),    g_j <- -eps log sum_i a_i exp((f_i - C_ij) / eps).

    Leading axes of C are independent problems with the same weights; returns f (..., n) and g (..., m). Every update
    forms ``(g_j - C_ij) / eps + log b_j`` in one work buffer of the shape of C (the same elementwise operations and
    log-sum-exp reduction as the out-of-place expression, without reallocating the temporaries).
    """
    f = C.new_zeros(C.shape[:-1])
    g = C.new_zeros(C.shape[:-2] + C.shape[-1:])
    T = torch.empty_like(C)
    la_col = la.unsqueeze(-1)
    for _ in range(n_iter):
        torch.sub(g.unsqueeze(-2), C, out=T)
        f = torch.logsumexp(T.div_(eps).add_(lb), dim=-1).mul_(-eps)
        torch.sub(f.unsqueeze(-1), C, out=T)
        g = torch.logsumexp(T.div_(eps).add_(la_col), dim=-2).mul_(-eps)
    return f, g


def _sym_potential(C: Tensor, la: Tensor, eps: float, n_iter: int) -> Tensor:
    """Symmetric potential of OT_eps(a, a) on the cost C (..., n, n) by the averaged iteration, started at 0,

        f_i <- 1/2 ( f_i - eps log sum_j a_j exp((f_j - C_ij) / eps) )   (one work buffer).
    """
    f = C.new_zeros(C.shape[:-1])
    T = torch.empty_like(C)
    for _ in range(n_iter):
        torch.sub(f.unsqueeze(-2), C, out=T)
        f = (f - torch.logsumexp(T.div_(eps).add_(la), dim=-1).mul_(eps)).mul_(0.5)
    return f


_GRAPH_CACHE_SIZE: int = 16
_graphs: OrderedDict[tuple, Any] = OrderedDict()


def _replay(kind: str, C: Tensor, la: Tensor, lb: Tensor | None, eps: float, n_iter: int) -> tuple[Tensor, ...]:
    """A potential loop on CUDA, run as a captured CUDA graph.

    The loop is captured once per (kind, shape, strides, dtype, device, eps, n_iter, calling thread) on static input
    buffers; a call copies its inputs into the buffers and replays the graph (the kernels of the eager loop with its
    launch parameters, so the same potentials bit for bit). At most ``_GRAPH_CACHE_SIZE`` graphs are kept (least
    recently used first out); if a capture fails, the loop runs eagerly for that key.
    """
    key = (kind, tuple(C.shape), C.stride(), C.dtype, C.device, float(eps), int(n_iter), threading.get_ident())
    entry = _graphs.get(key)
    if entry is None:
        entry = _capture(kind, C, la, lb, eps, n_iter)
        _graphs[key] = entry
        if len(_graphs) > _GRAPH_CACHE_SIZE:
            _graphs.popitem(last=False)
    else:
        _graphs.move_to_end(key)
    if entry is False:
        return (_cross_potentials(C, la, lb, eps, n_iter) if kind == "cross"
                else (_sym_potential(C, la, eps, n_iter),))
    graph, inputs, outputs = entry
    with torch.cuda.device(C.device):
        for buf, x in zip(inputs, (C, la, lb)):
            buf.copy_(x)
        graph.replay()
        return tuple(o.clone() for o in outputs)


def _capture(kind: str, C: Tensor, la: Tensor, lb: Tensor | None, eps: float, n_iter: int) -> Any:
    """Capture the potential loop of ``kind`` on static copies of the inputs: ``(graph, inputs, outputs)``, or False
    (with a logged warning) when the capture fails."""
    inputs = tuple(x.detach().clone() for x in ((C, la, lb) if lb is not None else (C, la)))

    def run() -> tuple[Tensor, ...]:
        if kind == "cross":
            return _cross_potentials(inputs[0], inputs[1], inputs[2], eps, n_iter)
        return (_sym_potential(inputs[0], inputs[1], eps, n_iter),)

    try:
        with torch.cuda.device(C.device):                           # the capture stream lives on the tensors' device
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                run()                                                # warm-up outside the capture
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                outputs = run()
    except RuntimeError as exc:                                      # e.g. no caching allocator
        logger.warning("CUDA graph capture of the Sinkhorn potentials failed (%s); running them eagerly", exc)
        return False
    return graph, inputs, outputs


def _cross(C: Tensor, la: Tensor, lb: Tensor, eps: float, n_iter: int) -> tuple[Tensor, Tensor]:
    """`_cross_potentials`, replayed from a CUDA graph for CUDA tensors (same values)."""
    if C.is_cuda:
        return _replay("cross", C, la, lb, eps, n_iter)  # type: ignore[return-value]
    return _cross_potentials(C, la, lb, eps, n_iter)


def _sym(C: Tensor, la: Tensor, eps: float, n_iter: int) -> Tensor:
    """`_sym_potential`, replayed from a CUDA graph for CUDA tensors (same values)."""
    if C.is_cuda:
        return _replay("sym", C, la, None, eps, n_iter)[0]
    return _sym_potential(C, la, eps, n_iter)


def _refined_value(Cxy: Tensor, Cxx: Tensor, Cyy: Tensor | None, a: Tensor, b: Tensor, la: Tensor, lb: Tensor,
                   f: Tensor, g: Tensor, f_aa: Tensor, g_bb: Tensor | None, eps: float) -> Tensor:
    """One differentiable iteration from the fixed potentials and the debiased value of one frame,

        S = <a, F_ab> + <b, G_ab> - <a, F_aa> [- <b, G_bb>],

    with F_ab, G_ab, F_aa, G_bb the Sinkhorn updates of (g, f, f_aa, g_bb) on the differentiable costs. The bracketed
    y-y term is omitted when ``Cyy`` is None (it does not depend on the generated points).
    """
    F_ab = -eps * torch.logsumexp(lb[None, :] + (g[None, :] - Cxy) / eps, dim=1)
    G_ab = -eps * torch.logsumexp(la[:, None] + (f[:, None] - Cxy) / eps, dim=0)
    F_aa = -eps * torch.logsumexp(la[None, :] + (f_aa[None, :] - Cxx) / eps, dim=1)
    G_bb = -eps * torch.logsumexp(lb[None, :] + (g_bb[None, :] - Cyy) / eps, dim=1) if Cyy is not None else None
    value = (a * F_ab).sum() + (b * G_ab).sum() - (a * F_aa).sum()
    if G_bb is not None:
        value = value - (b * G_bb).sum()
    return value


def _same_shapes(xs: Sequence[Tensor], ys: Sequence[Tensor]) -> bool:
    x0, y0 = xs[0], ys[0]
    return (all(x.shape == x0.shape and x.dtype == x0.dtype and x.device == x0.device for x in xs)
            and all(y.shape == y0.shape and y.dtype == y0.dtype and y.device == y0.device for y in ys))


# ------------------------------------------------------------------------------------------------ Sinkhorn
def sinkhorn_divergence(x: Tensor, y: Tensor, eps: float, n_iter: int = 100, *,
                        with_constant: bool = True) -> Tensor:
    """Debiased Sinkhorn divergence S_eps between the uniform measures on the rows of x (n, d) and y (m, d).

    Args:
        x: generated points (differentiated); y: observed points.
        eps: entropic regularisation (squared-distance units); n_iter: iterations of every potential.
        with_constant: include the y-y term -1/2 OT_eps(beta, beta). Without it the value changes by a constant in x
            and the gradient with respect to x is the same; the trainer evaluates it only at logging steps.

    Returns:
        0-dim tensor; its gradient with respect to x is that of 2 S_eps at converged potentials.
    """
    n, m = x.shape[0], y.shape[0]
    a, b = _uniform(n, x), _uniform(m, y)
    la, lb = _log_weights(a), _log_weights(b)
    Cxy = _sqdist(x, y)
    Cxx = _sqdist(x, x)
    Cyy = _sqdist(y, y) if with_constant else None
    with torch.no_grad():
        f, g = _cross(Cxy.detach(), la, lb, eps, n_iter)
        f_aa = _sym(Cxx.detach(), la, eps, n_iter)
        g_bb = _sym(Cyy.detach(), lb, eps, n_iter) if Cyy is not None else None
    return _refined_value(Cxy, Cxx, Cyy, a, b, la, lb, f, g, f_aa, g_bb, eps)


def sinkhorn_divergence_frames(xs: Sequence[Tensor], ys: Sequence[Tensor], eps: float, n_iter: int = 100, *,
                               with_constant: bool = True, batched: bool = True) -> list[Tensor]:
    """``sinkhorn_divergence(xs[k], ys[k], eps, n_iter)`` for every frame k, in frame order.

    With ``batched=True`` and frames of equal shapes, the detached cost matrices of all frames are stacked to (F, n, m)
    and the potential iterations run once on the stack (the same expressions with a leading frame axis); every frame
    then gets its own differentiable refinement. On the CPU the result equals the per-frame loop bit for bit; on CUDA
    only to rounding, so the trainer batches on the CPU and runs frame by frame on CUDA.
    """
    if len(xs) != len(ys):
        raise ValueError(f"xs and ys must have the same number of frames, got {len(xs)} and {len(ys)}")
    if len(xs) == 0:
        return []
    if not batched or not _same_shapes(xs, ys):
        return [sinkhorn_divergence(x, y, eps, n_iter, with_constant=with_constant) for x, y in zip(xs, ys)]
    n, m = xs[0].shape[0], ys[0].shape[0]
    a, b = _uniform(n, xs[0]), _uniform(m, ys[0])
    la, lb = _log_weights(a), _log_weights(b)
    Cxy = [_sqdist(x, y) for x, y in zip(xs, ys)]
    Cxx = [_sqdist(x, x) for x in xs]
    Cyy = [_sqdist(y, y) for y in ys] if with_constant else None
    with torch.no_grad():
        f, g = _cross(torch.stack([c.detach() for c in Cxy]), la, lb, eps, n_iter)   # (F, n), (F, m)
        f_aa = _sym(torch.stack([c.detach() for c in Cxx]), la, eps, n_iter)          # (F, n)
        g_bb = (_sym(torch.stack([c.detach() for c in Cyy]), lb, eps, n_iter)         # (F, m)
                if Cyy is not None else None)
    return [_refined_value(Cxy[k], Cxx[k], None if Cyy is None else Cyy[k], a, b, la, lb, f[k], g[k], f_aa[k],
                           None if g_bb is None else g_bb[k], eps)
            for k in range(len(xs))]


# ----------------------------------------------------------------------------------------------------- MMD
def rbf_mmd2(x: Tensor, y: Tensor, sigmas: Sequence[float], *, with_constant: bool = True) -> Tensor:
    """Multi-bandwidth Gaussian MMD^2 between the uniform measures on the rows of x (n, d) and y (m, d),

        MMD^2 = (1/S) sum_s [ a^T K_s(x, x) a + b^T K_s(y, y) b - 2 a^T K_s(x, y) b ],
        K_s(u, v)_ij = exp(-|u_i - v_j|^2 / (2 s^2)).

    ``with_constant=False`` drops the y-y term (constant in x; same gradient). Returns a 0-dim tensor.
    """
    if len(sigmas) == 0:
        raise ValueError("rbf_mmd2 needs at least one bandwidth")
    n, m = x.shape[0], y.shape[0]
    a, b = _uniform(n, x), _uniform(m, y)
    Cxx = _sqdist(x, x)
    Cyy = _sqdist(y, y) if with_constant else None
    Cxy = _sqdist(x, y)
    tot = 0.0
    for s in sigmas:
        gam = 1.0 / (2.0 * s * s)
        kxx = torch.exp(-gam * Cxx)
        if Cyy is not None:
            kyy = torch.exp(-gam * Cyy)
            kxy = torch.exp(-gam * Cxy)
            tot = tot + (a @ kxx @ a) + (b @ kyy @ b) - 2.0 * (a @ kxy @ b)
        else:
            kxy = torch.exp(-gam * Cxy)
            tot = tot + (a @ kxx @ a) - 2.0 * (a @ kxy @ b)
    return tot / len(sigmas)


def median_sigmas(X: np.ndarray, seed: int = 0, n_pairs: int = 4000) -> tuple[float, ...]:
    """MMD bandwidths ``MMD_MULTIPLIERS x median |X_i - X_j|``.

    The median runs over ``n_pairs`` random pairs drawn with replacement from ``numpy.random.default_rng(seed)``
    (all i first, then all j) and over the nonzero distances only; it is 1.0 when every sampled distance is zero.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(X), size=n_pairs)
    j = rng.integers(0, len(X), size=n_pairs)
    d = np.linalg.norm(X[i] - X[j], axis=1)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
    return tuple(float(m * med) for m in MMD_MULTIPLIERS)


def stratified_times(n: int, generator: torch.Generator | None = None,
                     device: str | torch.device = "cpu") -> Tensor:
    """Stratified uniform times on [0, 1]: ``t_i = (i + u_i) / n`` with ``u_i ~ U[0, 1)``, shape (n,), float64.

    One draw per stratum (``torch.rand(n, generator=generator)``), so a time integral is estimated with lower
    variance than with n independent uniform times.
    """
    return (torch.arange(n, device=device, dtype=torch.float64)
            + torch.rand(n, generator=generator, device=device, dtype=torch.float64)) / n


# -------------------------------------------------------------------------------------------- auto settings
def auto_sinkhorn_eps(frames: Sequence[np.ndarray]) -> float:
    """Sinkhorn eps of the retained snapshots: ``max(1e-3, 0.02 mean_k sum_j Var(frame_k[:, j]))`` (population
    variance), so eps scales with the squared-distance scale of the data."""
    if len(frames) == 0:
        raise ValueError("auto_sinkhorn_eps needs at least one frame")
    return max(1e-3, 0.02 * float(np.mean([np.var(np.asarray(c, dtype=np.float64), axis=0).sum() for c in frames])))


def auto_b_fit(frames: Sequence[np.ndarray], cap: int = 128) -> int:
    """Fit batch per snapshot: ``min(cap, smallest retained snapshot)``."""
    if len(frames) == 0:
        raise ValueError("auto_b_fit needs at least one frame")
    return int(min(cap, min(len(c) for c in frames)))
