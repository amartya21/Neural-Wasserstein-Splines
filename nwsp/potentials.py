"""Scalar potentials of the weak tangent projection: the fixed dictionary and the learned potentials of NWSP-N.

A dictionary of M potentials f_1, ..., f_M on R^d enters the projection only through its gradients and
Hessian-vector products, for points X (B, d) and directions Q (B, d):

    grad(X)       -> (B, d, M),  column m = grad f_m(x_b)          F(x) = [grad f_1(x) ... grad f_M(x)] in R^{d x M}
    hessvec(X, Q) -> (B, d, M),  column m = Hess f_m(x_b) q_b      (D_x F[q] in the projection)

The fixed dictionary concatenates, in this order,

    linear     f(x) = x_m,                m = 1..d          grad f = e_m,                Hess f q = 0
    quadratic  f(x) = x_i x_j,            i <= j            grad f = x_j e_i + x_i e_j,  Hess f q = q_j e_i + q_i e_j
    ridge      f(x) = s(w . x + b),       s = sin, cos      grad f = s'(a) w,            Hess f q = s''(a) (w . q) w

with a = w . x + b, Gaussian directions w ~ N(0, (fs / med)^2 I) for the frequency scales fs in (0.5, 1, 2) (med: the
median pairwise distance of the training points) and offsets b = -w . x_c that centre each ridge at a random training
point x_c; the sine and cosine blocks share (w, b). Every quadratic and ridge feature carries a constant scale, set on
the training points so that E|grad f_m|^2 = 1 (this reconditions the Gram matrix without changing the span).

The learned block of NWSP-N holds K potentials given by one tanh MLP with K outputs,

    g_k(x) = scale_k (h^L(x) W_o)_k,    h^0 = (x - m) / ell,    h^l = tanh(W_l h^{l-1} + b_l),    L in {1, 2},

with the centre m and length ell of the training data. Its Jacobian and the derivative of the Jacobian along q are
propagated in closed form (no autograd over x): with a_1 = W_1 h^0 + b_1 and da_1 = W_1 q / ell,

    J_1 = tanh'(a_1) W_1 / ell,    dJ_1 = tanh''(a_1) da_1 W_1 / ell,    dh_1 = tanh'(a_1) da_1,

and for a second layer, a_2 = W_2 h^1 + b_2, da_2 = W_2 dh_1,

    J_2 = tanh'(a_2) (W_2 J_1),    dJ_2 = tanh''(a_2) da_2 (W_2 J_1) + tanh'(a_2) (W_2 dJ_1);

then grad g = J^T W_o and Hess g q = dJ^T W_o (times scale).

Numerics: float64; the expressions and their operation order are kept fixed (training is reproducible bit for bit).
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .model import ArrayLike

logger = logging.getLogger(__name__)

FREQ_SCALES: tuple[float, ...] = (0.5, 1.0, 2.0)
"""Frequency scales of the ridge directions, in units of 1 / (median pairwise distance)."""

_RIDGE_KINDS: tuple[str, ...] = ("sin", "cos")
_ENERGY_FLOOR: float = 1e-30


class FeatureBlock(nn.Module):
    """A block of M scalar potentials on R^d with batched gradients and Hessian-vector products.

    A block with ``normalizable = True`` carries a ``scale`` buffer (M,) and implements ``feature_energy(X)``
    (E|grad f_m|^2 at the current scale) and ``set_scale(s)``, used by `Dictionary.normalize`.
    """

    d: int
    M: int
    normalizable: bool = False

    def grad(self, X: Tensor) -> Tensor:
        """(B, d, M): column m is grad f_m(x_b)."""
        raise NotImplementedError

    def hessvec(self, X: Tensor, Q: Tensor) -> Tensor:
        """(B, d, M): column m is Hess f_m(x_b) q_b."""
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """Plain description from which a block of the same shapes is rebuilt."""
        raise NotImplementedError


def _copy_scale(buffer: Tensor, s: Tensor | np.ndarray) -> None:
    value = torch.as_tensor(s, dtype=torch.float64)
    if value.shape != buffer.shape:
        raise ValueError(f"scale must have shape {tuple(buffer.shape)}, got {tuple(value.shape)}")
    buffer.copy_(value)


class LinearBlock(FeatureBlock):
    """Coordinate potentials f_m(x) = x_m: grad f_m = e_m, Hess f_m = 0. Buffer ``eye`` (d, d)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        if int(d) < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.d = int(d)
        self.M = self.d
        self.register_buffer("eye", torch.eye(self.d, dtype=torch.float64))

    def grad(self, X: Tensor) -> Tensor:
        """(B, d, d): the identity for every sample (an expanded view; do not modify it in place)."""
        return self.eye.expand(X.shape[0], self.d, self.d)

    def hessvec(self, X: Tensor, Q: Tensor) -> Tensor:
        return torch.zeros(X.shape[0], self.d, self.M, dtype=X.dtype, device=X.device)

    def spec(self) -> dict[str, Any]:
        return {"kind": "linear", "d": self.d}


class QuadraticBlock(FeatureBlock):
    """Quadratic potentials f(x) = x_i x_j, i <= j (upper triangle, row-major): grad f = x_j e_i + x_i e_j and
    Hess f q = q_j e_i + q_i e_j. Buffers ``ii``, ``jj`` (M,) and ``scale`` (M,), M = d (d + 1) / 2."""

    normalizable = True

    def __init__(self, d: int) -> None:
        super().__init__()
        if int(d) < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        self.d = int(d)
        iu = np.triu_indices(self.d)
        self.register_buffer("ii", torch.as_tensor(iu[0], dtype=torch.long))
        self.register_buffer("jj", torch.as_tensor(iu[1], dtype=torch.long))
        self.M = len(iu[0])
        self.register_buffer("scale", torch.ones(self.M, dtype=torch.float64))

    def _mat(self, A: Tensor, rows_i: Tensor, rows_j: Tensor) -> Tensor:
        """(B, d, M): ``rows_j[:, m]`` added at row ii[m], then ``rows_i[:, m]`` at row jj[m] of column m."""
        B, d = A.shape
        out = torch.zeros(B, d, self.M, dtype=A.dtype, device=A.device)
        out.scatter_add_(1, self.ii.view(1, 1, -1).expand(B, 1, self.M), rows_j.unsqueeze(1))
        out.scatter_add_(1, self.jj.view(1, 1, -1).expand(B, 1, self.M), rows_i.unsqueeze(1))
        return out

    def grad(self, X: Tensor) -> Tensor:
        return self._mat(X, X[:, self.ii], X[:, self.jj]) * self.scale

    def hessvec(self, X: Tensor, Q: Tensor) -> Tensor:
        return self._mat(X, Q[:, self.ii], Q[:, self.jj]) * self.scale

    def feature_energy(self, X: Tensor) -> Tensor:
        """(M,) E|grad f_m|^2 at the current scale, in closed form: per sample ``(x_j s)^2 + (x_i s)^2`` (i < j) or
        ``((x_i + x_i) s)^2`` (i = j), equal to ``(grad(X) ** 2).sum(1).mean(0)`` without the (B, d, M) tensor."""
        s = self.scale
        xi, xj = X[:, self.ii], X[:, self.jj]
        E = torch.where(self.ii == self.jj, ((xi + xi) * s) ** 2, (xj * s) ** 2 + (xi * s) ** 2)
        return E.mean(0)

    def set_scale(self, s: Tensor | np.ndarray) -> None:
        _copy_scale(self.scale, s)

    def spec(self) -> dict[str, Any]:
        return {"kind": "quadratic", "d": self.d, "M": self.M}


class RidgeBlock(FeatureBlock):
    """Ridge potentials f_m(x) = s(w_m . x + b_m), s = sin or cos for the whole block: grad f_m = s'(a_m) w_m and
    Hess f_m q = s''(a_m) (w_m . q) w_m (rank one). Buffers ``W`` (M, d), ``b`` (M,), ``scale`` (M,)."""

    normalizable = True

    def __init__(self, W: np.ndarray | Tensor, b: np.ndarray | Tensor, kind: str = "sin") -> None:
        super().__init__()
        if kind not in _RIDGE_KINDS:
            raise ValueError(f"kind must be one of {_RIDGE_KINDS}, got {kind!r}")
        W_t = torch.as_tensor(W, dtype=torch.float64).detach().clone()
        if W_t.ndim != 2:
            raise ValueError(f"W must have shape (M, d), got {tuple(W_t.shape)}")
        b_t = torch.as_tensor(b, dtype=torch.float64).detach().clone().to(W_t.device)
        if b_t.shape != (W_t.shape[0],):
            raise ValueError(f"b must have shape ({W_t.shape[0]},), got {tuple(b_t.shape)}")
        self.kind = kind
        self.M = int(W_t.shape[0])
        self.d = int(W_t.shape[1])
        self.register_buffer("W", W_t)
        self.register_buffer("b", b_t)
        self.register_buffer("scale", torch.ones(self.M, dtype=torch.float64, device=W_t.device))

    def _a(self, X: Tensor) -> Tensor:
        return X @ self.W.T + self.b

    def grad(self, X: Tensor) -> Tensor:
        a = self._a(X)
        d1 = torch.cos(a) if self.kind == "sin" else -torch.sin(a)            # s'(a), (B, M)
        Wt = self.W.T.unsqueeze(0)                                             # (1, d, M)
        return Wt * (d1 * self.scale).unsqueeze(1)

    def hessvec(self, X: Tensor, Q: Tensor) -> Tensor:
        a = self._a(X)
        d2 = -torch.sin(a) if self.kind == "sin" else -torch.cos(a)           # s''(a), (B, M)
        wq = Q @ self.W.T                                                      # w_m . q_b, (B, M)
        Wt = self.W.T.unsqueeze(0)
        return Wt * (d2 * wq * self.scale).unsqueeze(1)

    def feature_energy(self, X: Tensor) -> Tensor:
        """(M,) E|grad f_m|^2 at the current scale."""
        return (self.grad(X) ** 2).sum(1).mean(0)

    def set_scale(self, s: Tensor | np.ndarray) -> None:
        _copy_scale(self.scale, s)

    def spec(self) -> dict[str, Any]:
        return {"kind": "ridge", "d": self.d, "M": self.M, "activation": self.kind}


class Dictionary(nn.Module):
    """Concatenation of feature blocks, F(x) = [F_1(x) | F_2(x) | ...] in R^{d x M} (block order kept)."""

    def __init__(self, blocks: Sequence[FeatureBlock]) -> None:
        super().__init__()
        blocks = list(blocks)
        if not blocks:
            raise ValueError("a Dictionary needs at least one block")
        dims = {int(blk.d) for blk in blocks}
        if len(dims) != 1:
            raise ValueError(f"all blocks must have the same input dimension d, got {sorted(dims)}")
        self.blocks = nn.ModuleList(blocks)
        self.d = dims.pop()
        self.M = sum(int(blk.M) for blk in blocks)

    def grad(self, X: Tensor) -> Tensor:
        """(B, d, M) gradients of all potentials."""
        return torch.cat([blk.grad(X) for blk in self.blocks], dim=2)

    def hessvec(self, X: Tensor, Q: Tensor) -> Tensor:
        """(B, d, M) Hessian-vector products Hess f_m(x_b) q_b."""
        return torch.cat([blk.hessvec(X, Q) for blk in self.blocks], dim=2)

    @torch.no_grad()
    def normalize(self, X: Tensor) -> None:
        """Rescale every normalizable block so that E_X |grad f_m|^2 = 1: ``scale <- scale / sqrt(max(E, 1e-30))``."""
        for blk in self.blocks:
            if blk.normalizable:
                energy = blk.feature_energy(X).clamp_min(_ENERGY_FLOOR)
                blk.set_scale(blk.scale / energy.sqrt())

    def spec(self) -> list[dict[str, Any]]:
        """Block specs in block order (rebuild with `from_spec`, then load the state dict)."""
        return [blk.spec() for blk in self.blocks]

    @classmethod
    def from_spec(cls, specs: Sequence[Mapping[str, Any]]) -> Dictionary:
        """Fixed blocks of the shapes of ``specs`` (placeholder buffers); load the saved state dict afterwards."""
        blocks: list[FeatureBlock] = []
        for s in specs:
            if s["kind"] == "linear":
                blocks.append(LinearBlock(int(s["d"])))
            elif s["kind"] == "quadratic":
                blocks.append(QuadraticBlock(int(s["d"])))
            elif s["kind"] == "ridge":
                blocks.append(RidgeBlock(np.zeros((int(s["M"]), int(s["d"]))), np.zeros(int(s["M"])),
                                         kind=s["activation"]))
            else:
                raise ValueError(f"unknown fixed block kind {s['kind']!r}")
            if "M" in s and int(s["M"]) != blocks[-1].M:
                raise ValueError(f"spec {dict(s)} declares M = {s['M']}, the rebuilt block has M = {blocks[-1].M}")
        return cls(blocks)

    def extra_repr(self) -> str:
        return f"d={self.d}, M={self.M}"


def _median_distance(X: np.ndarray, rng: np.random.Generator, n_pairs: int = 2000) -> float:
    """Median Euclidean distance of random pairs of rows of X (all i, then all j; nonzero distances only; 1.0 if
    there are none), floored at 1e-8."""
    n = len(X)
    i = rng.integers(0, n, size=min(n_pairs, n * n))
    j = rng.integers(0, n, size=len(i))
    dist = np.linalg.norm(X[i] - X[j], axis=1)
    med = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 1.0
    return max(med, 1e-8)


def build_dictionary(X: np.ndarray, n_ridge: int = 32, seed: int = 0,
                     device: str | torch.device = "cpu") -> Dictionary:
    """The fixed dictionary of the training points X (N, d), normalised on X.

    Blocks: linear (d), quadratic (d (d + 1) / 2) and, if n_ridge > 0, a sine and a cosine ridge block sharing
    ``3 ceil(n_ridge / 6)`` directions (6 ceil(n_ridge / 6) ridge features: 36 for n_ridge = 32). Draws from
    ``np.random.default_rng(seed)`` in this order: the pairs of the median distance, then per frequency scale fs the
    directions ``normal(size=(per, d)) * (fs / med)`` and the centre indices ``integers(0, N, size=per)``.
    """
    X = np.asarray(X)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"X must be a non-empty (N, d) array, got shape {X.shape}")
    d = X.shape[1]
    rng = np.random.default_rng(seed)
    blocks: list[FeatureBlock] = [LinearBlock(d), QuadraticBlock(d)]
    if n_ridge > 0:
        med = _median_distance(X, rng)
        per = math.ceil(n_ridge / (2 * len(FREQ_SCALES)))
        Ws, bs = [], []
        for fs in FREQ_SCALES:
            W = rng.normal(size=(per, d)) * (fs / med)
            centers = X[rng.integers(0, len(X), size=per)]
            Ws.append(W)
            bs.append(-(W * centers).sum(1))
        W, b = np.concatenate(Ws), np.concatenate(bs)
        blocks.append(RidgeBlock(W, b, kind="sin"))
        blocks.append(RidgeBlock(W, b, kind="cos"))
    dic = Dictionary(blocks).to(device)
    dic.normalize(torch.as_tensor(X, dtype=torch.float64, device=device))
    logger.debug("fixed dictionary: d=%d, M=%d, block sizes %s", d, dic.M, [blk.M for blk in blocks])
    return dic


# ------------------------------------------------------------------------------------------- learned potentials
def _tanh_derivs(a: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(tanh(a), tanh'(a), tanh''(a))`` = ``(t, 1 - t^2, -2 t (1 - t^2))`` elementwise."""
    t = torch.tanh(a)
    s1 = 1.0 - t * t
    return t, s1, -2.0 * t * s1


class NeuralPotentialBlock(FeatureBlock):
    """K learned scalar potentials of x (a tanh MLP) with closed-form gradients and Hessian-vector products.

    Parameters, registered in this order: ``W1`` (w, d), ``b1`` (w), [``W2`` (w, w), ``b2`` (w) for depth 2],
    ``Wo`` (w, K); buffers ``m`` (d,), ``ell`` (), ``scale`` (K,). The initial weights are drawn in float64 from
    ``torch.Generator().manual_seed(seed)`` in this order:

        W1 = randn(w, d) init_gamma,  b1 = randn(w) 0.5,  W2 = randn(w, w) / sqrt(w),  b2 = randn(w) 0.1,
        Wo = randn(w, K) / sqrt(w).

    The block is built on the CPU (move it with ``.to(device)``); its scale is set by `refresh_scale`.

    Args:
        d: dimension. K: number of potentials. width, depth: hidden width and number of hidden layers (1 or 2).
        center, length: the input map h^0 = (x - center) / length (default: 0 and 1).
        seed: seed of the initial weights. init_gamma: scale of the first-layer weights.
    """

    def __init__(self, d: int, K: int = 16, width: int = 64, depth: int = 2,
                 center: ArrayLike | Tensor | None = None, length: float = 1.0, seed: int = 0,
                 init_gamma: float = 1.5) -> None:
        super().__init__()
        if int(K) <= 0:
            raise ValueError(f"K must be positive, got {K}")
        if int(d) < 1 or int(width) < 1:
            raise ValueError(f"d and width must be >= 1, got d={d}, width={width}")
        if depth not in (1, 2):
            raise ValueError(f"depth must be 1 or 2, got {depth}")
        if not (math.isfinite(float(length)) and float(length) > 0.0):
            raise ValueError(f"length must be finite and positive, got {length}")
        self.d, self.K, self.M = int(d), int(K), int(K)
        self.width, self.depth = int(width), int(depth)

        g = torch.Generator().manual_seed(int(seed))
        w, f64 = self.width, torch.float64
        self.W1 = nn.Parameter(torch.randn(w, self.d, generator=g, dtype=f64) * float(init_gamma))
        self.b1 = nn.Parameter(torch.randn(w, generator=g, dtype=f64) * 0.5)
        if self.depth == 2:
            self.W2 = nn.Parameter(torch.randn(w, w, generator=g, dtype=f64) * (1.0 / math.sqrt(w)))
            self.b2 = nn.Parameter(torch.randn(w, generator=g, dtype=f64) * 0.1)
        self.Wo = nn.Parameter(torch.randn(w, self.K, generator=g, dtype=f64) * (1.0 / math.sqrt(w)))

        m = torch.zeros(self.d, dtype=f64) if center is None else torch.as_tensor(center, dtype=f64, device="cpu")
        if m.shape != (self.d,):
            raise ValueError(f"center must have shape ({self.d},), got {tuple(m.shape)}")
        self.register_buffer("m", m.detach().clone())
        self.register_buffer("ell", torch.tensor(float(length), dtype=f64))
        self.register_buffer("scale", torch.ones(self.K, dtype=f64))

    def extra_repr(self) -> str:
        return f"d={self.d}, K={self.K}, width={self.width}, depth={self.depth}"

    def _pass(self, X: Tensor, Q: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """(J, dJ): the (B, w, d) Jacobian of the last hidden layer and its derivative along Q (None without Q)."""
        ell = self.ell
        xt = (X - self.m) / ell
        W1e = (self.W1 / ell).unsqueeze(0)                                       # (1, w, d)
        a1 = xt @ self.W1.T + self.b1
        h1, s1, c1 = _tanh_derivs(a1)
        J = s1.unsqueeze(-1) * W1e                                               # (B, w, d)
        dJ = None
        if Q is not None:
            da1 = (Q / ell) @ self.W1.T                                          # (B, w)
            dJ = (c1 * da1).unsqueeze(-1) * W1e
            dh = s1 * da1
        if self.depth == 2:
            a2 = h1 @ self.W2.T + self.b2
            _, s2, c2 = _tanh_derivs(a2)
            WJ = torch.einsum("ij,bjd->bid", self.W2, J)                         # (B, w, d)
            J2 = s2.unsqueeze(-1) * WJ
            if Q is not None:
                da2 = dh @ self.W2.T
                dWJ = torch.einsum("ij,bjd->bid", self.W2, dJ)
                dJ = (c2 * da2).unsqueeze(-1) * WJ + s2.unsqueeze(-1) * dWJ
            J = J2
        return J, dJ

    def _out(self, J: Tensor) -> Tensor:
        """(B, d, K) = J^T W_o per sample."""
        return torch.einsum("bwd,wk->bdk", J, self.Wo)

    def grad_raw(self, X: Tensor) -> Tensor:
        """(B, d, K) gradients without the scale."""
        J, _ = self._pass(X, None)
        return self._out(J)

    def hessvec_raw(self, X: Tensor, Q: Tensor) -> Tensor:
        """(B, d, K) Hessian-vector products Hess g_k(x_b) q_b without the scale (own forward pass)."""
        _, dJ = self._pass(X, Q)
        return self._out(dJ)

    def grad(self, X: Tensor) -> Tensor:
        return self.grad_raw(X) * self.scale

    def hessvec(self, X: Tensor, Q: Tensor) -> Tensor:
        return self.hessvec_raw(X, Q) * self.scale

    @torch.no_grad()
    def raw_energy(self, X: Tensor) -> Tensor:
        """(K,) E_X |grad_raw g_k|^2 (without the scale)."""
        G = self.grad_raw(X)
        return (G ** 2).sum(1).mean(0)

    @torch.no_grad()
    def refresh_scale(self, X: Tensor, floor: float = 1e-12) -> Tensor:
        """Set ``scale_k = 1 / sqrt(max(E_X |grad_raw g_k|^2, floor))`` (so E_X |grad g_k|^2 = 1); returns it."""
        e = self.raw_energy(X).clamp_min(floor)
        self.scale.copy_(1.0 / e.sqrt())
        return self.scale

    def spec(self) -> dict[str, Any]:
        return {"kind": "neural", "d": self.d, "K": self.K, "width": self.width, "depth": self.depth}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> NeuralPotentialBlock:
        """A block of the shapes of ``spec`` (seed 0); load the saved state dict afterwards."""
        if spec.get("kind", "neural") != "neural":
            raise ValueError(f"not a neural block spec: {dict(spec)}")
        return cls(int(spec["d"]), int(spec["K"]), int(spec["width"]), int(spec["depth"]), seed=0)
