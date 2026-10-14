"""Spline generator: a neural network maps a latent variable to the control points of a cubic B-spline in time.

Time is normalised so that the K retained snapshot times become the knots 0 = t_0 < t_1 < ... < t_{K-1} = 1. The
clamped knot vector repeats each end time four times and keeps the interior times as simple knots,

    xi = (t_0, t_0, t_0, t_0, t_1, ..., t_{K-2}, t_{K-1}, t_{K-1}, t_{K-1}, t_{K-1}),

and carries J = K + 2 cubic B-splines, given by the Cox-de Boor recursion (0/0 := 0)

    B_{j,0}(t) = 1[xi_j <= t < xi_{j+1}],
    B_{j,p}(t) = (t - xi_j) / (xi_{j+p} - xi_j) B_{j,p-1}(t)
                 + (xi_{j+p+1} - t) / (xi_{j+p+1} - xi_{j+1}) B_{j+1,p-1}(t),

with time derivatives d/dt B_{j,p} = p [B_{j,p-1} / (xi_{j+p} - xi_j) - B_{j+1,p-1} / (xi_{j+p+1} - xi_{j+1})].
Times outside [0, 1] evaluate the cubic of the first / last span.

The generator is

    G(t, z) = sum_j B_j(t) g_j(z),        g(z) = trunk(z) + skip(z)  reshaped to (J, d),

with an MLP trunk (SiLU, last layer initialised at zero) and an affine skip. A sample at time t is X_t = G(t, Z) with
Z ~ N(0, I) held fixed along a path, so one network evaluation gives a sample at any time; U = dX/dt and
W = d^2X/dt^2 (normalised time) are the velocity and acceleration of this representation. Initialisation
(`SplineGenerator.init_from_data`): the control points of z = 0 are the least-squares fit of the snapshot means with
natural end conditions G''(0) = G''(1) = 0, and the skip weight of every control point is the Cholesky factor of the
pooled snapshot covariance (its first dz columns).

Numerics: float64 throughout. The basis is computed with numpy (elementwise IEEE arithmetic) and cached per query-time
vector; the layers are drawn from a seeded copy of the global CPU generator, which is left unchanged.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray | Sequence[float]

DEGREE: int = 3
"""Spline degree (cubic)."""
SAMPLE_CHUNK: int = 100_000
"""Particles per generator evaluation when sampling."""

_ORDERS: tuple[int, ...] = (0, 1, 2)
_COV_JITTER: float = 1e-8        # added to the diagonal of the pooled covariance before its Cholesky factor
_COV_REL_JITTER: float = 1e-10   # fallback, relative to the mean variance, for a numerically singular covariance


# ----------------------------------------------------------------------------------------------------- basis
def clamped_knots(times: ArrayLike) -> np.ndarray:
    """Clamped knot vector ``(t_0 x4, t_1, ..., t_{K-2}, t_{K-1} x4)`` of shape (K + 6,).

    Raises:
        ValueError: unless ``times`` is 1-D with K >= 2 finite, strictly increasing entries.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("times must be a 1-D array with at least 2 entries")
    if not np.all(np.isfinite(t)):
        raise ValueError("times must be finite")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    return np.concatenate([np.full(DEGREE + 1, t[0]), t[1:-1], np.full(DEGREE + 1, t[-1])])


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num / den`` over the last axis with 0 wherever ``den == 0`` (0/0 := 0)."""
    out = np.zeros_like(num)
    nz = den != 0.0
    out[..., nz] = num[..., nz] / den[nz]
    return out


def _degree0(t: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """(Q, len(knots) - 1) indicators of the spans ``[xi_j, xi_{j+1})``; times at or after the last knot use the last
    non-empty span and times before the first knot the first one (end polynomials)."""
    lo, hi = knots[:-1], knots[1:]
    N0 = ((t[:, None] >= lo[None, :]) & (t[:, None] < hi[None, :])).astype(np.float64)
    last_span = int(np.max(np.nonzero(hi > lo)[0]))
    at_end = t >= knots[-1]
    if np.any(at_end):
        N0[at_end, :] = 0.0
        N0[at_end, last_span] = 1.0
    first_span = int(np.min(np.nonzero(hi > lo)[0]))
    before = t < knots[0]
    if np.any(before):
        N0[before, :] = 0.0
        N0[before, first_span] = 1.0
    return N0


def _spans(knots: np.ndarray, p: int, n_p: int) -> tuple[np.ndarray, np.ndarray]:
    """Denominators ``xi_{j+p} - xi_j`` and ``xi_{j+p+1} - xi_{j+1}`` of the degree-p recursion, j < n_p."""
    d1 = knots[p : p + n_p] - knots[0:n_p]
    d2 = knots[p + 1 : p + 1 + n_p] - knots[1 : 1 + n_p]
    return d1, d2


def _raise_values(t: np.ndarray, knots: np.ndarray, Nprev: np.ndarray, p: int, n_p: int) -> np.ndarray:
    """Degree-p values (Q, n_p) from the degree-(p - 1) values (Cox-de Boor)."""
    d1, d2 = _spans(knots, p, n_p)
    left = _safe_div(t[:, None] - knots[0:n_p][None, :], d1) * Nprev[:, 0:n_p]
    right = _safe_div(knots[p + 1 : p + 1 + n_p][None, :] - t[:, None], d2) * Nprev[:, 1 : 1 + n_p]
    return left + right


def _raise_derivative(knots: np.ndarray, Dprev: np.ndarray, p: int, n_p: int) -> np.ndarray:
    """d/dt of a degree-p quantity (Q, n_p) from the degree-(p - 1) quantity of the same derivative order."""
    d1, d2 = _spans(knots, p, n_p)
    a = _safe_div(np.ones((1, n_p)), d1) * Dprev[:, 0:n_p]
    b = _safe_div(np.ones((1, n_p)), d2) * Dprev[:, 1 : 1 + n_p]
    return p * (a - b)


def basis_matrices(times: ArrayLike, knots: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic B-spline basis and its first two time derivatives at the query times.

    Returns:
        ``(N, dN, d2N)``, each (Q, J) float64: ``N[q, j] = B_j(t_q)``, ``dN = d/dt B_j``, ``d2N = d^2/dt^2 B_j``.
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    knots = np.asarray(knots, dtype=np.float64)
    vals = [_degree0(t, knots)]
    for p in range(1, DEGREE + 1):
        vals.append(_raise_values(t, knots, vals[p - 1], p, len(knots) - p - 1))
    d1_deg2 = _raise_derivative(knots, vals[1], 2, len(knots) - 3)   # first derivative of the degree-2 basis
    d1_deg3 = _raise_derivative(knots, vals[2], 3, len(knots) - 4)   # first derivative of the cubic basis
    d2_deg3 = _raise_derivative(knots, d1_deg2, 3, len(knots) - 4)   # second derivative of the cubic basis
    return vals[3], d1_deg3, d2_deg3


def _time_array(t: float | ArrayLike | Tensor) -> np.ndarray:
    """Query times as a flat float64 numpy array (tensors are read back to the host, values are not rounded)."""
    if torch.is_tensor(t):
        return t.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1)
    if isinstance(t, (list, tuple)) and any(torch.is_tensor(x) for x in t):
        return np.array([float(x) for x in t], dtype=np.float64)
    return np.asarray(t, dtype=np.float64).reshape(-1)


def _scalar_time(t: Any) -> float:
    """A single time (a number, a 0-dim or one-element array) as a python float; ValueError otherwise."""
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    arr = np.asarray(t, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"t must be a single time, got an array of shape {arr.shape}")
    return float(arr.reshape(()))


def _positive_int(value: Any, name: str) -> int:
    """``value`` as a python int; ValueError unless it is a whole number >= 1."""
    try:
        ok = not isinstance(value, bool) and int(value) == value and int(value) >= 1
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


class CubicBSpline:
    """Clamped cubic B-spline basis on fixed knot times, as float64 torch matrices at any query times.

    The matrices are kept in a least-recently-used cache keyed by the query values and the device (training evaluates
    the same few times at every step). Cached tensors are shared between calls: do not modify them in place.

    Args:
        knot_times: (K,) strictly increasing knot times, K >= 2.
        cache_size: number of query-time vectors kept (0 disables the cache).

    Attributes:
        knot_times: (K,) float64; knots: the clamped knot vector (K + 6,); J: number of basis functions (K + 2).
    """

    def __init__(self, knot_times: ArrayLike, cache_size: int = 64) -> None:
        if int(cache_size) < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")
        self.knot_times: np.ndarray = np.asarray(knot_times, dtype=np.float64)
        self.knots: np.ndarray = clamped_knots(self.knot_times)
        self.J: int = len(self.knots) - DEGREE - 1
        self.cache_size: int = int(cache_size)
        self._cache: OrderedDict[tuple, tuple[Tensor, Tensor, Tensor]] = OrderedDict()

    def __call__(self, t: float | ArrayLike | Tensor,
                 device: torch.device | str | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """``(N, dN, d2N)``, each (Q, J) float64, at the query times ``t`` (flattened to Q values).

        ``device`` defaults to the device of ``t`` if it is a tensor, else the CPU.
        """
        t_np = _time_array(t)
        if device is None:
            device = t.device if torch.is_tensor(t) else torch.device("cpu")
        device = torch.device(device)
        key = (t_np.tobytes(), t_np.shape[0], str(device))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        mats = basis_matrices(t_np, self.knots)
        with torch.inference_mode(False):      # ordinary tensors, also when first built under inference mode
            val = tuple(torch.as_tensor(a, dtype=torch.float64, device=device) for a in mats)
        if self.cache_size > 0:
            self._cache[key] = val
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return val

    def __repr__(self) -> str:
        return f"CubicBSpline(K={len(self.knot_times)}, J={self.J})"


# ------------------------------------------------------------------------------------------------- generator
class SplineGenerator(nn.Module):
    """Latent-to-control-point network evaluated through a clamped cubic B-spline basis.

    Args:
        d: data dimension.
        dz: latent dimension.
        knot_times: (K,) strictly increasing normalised knot times (the retained snapshot times); J = K + 2.
        width: hidden width of the trunk MLP.
        depth: number of hidden SiLU layers of the trunk MLP.
        seed: the layers are initialised from ``torch.manual_seed(seed)`` inside a forked CPU random state.

    Attributes:
        trunk: ``[Linear(dz, width), SiLU, (Linear(width, width), SiLU) x (depth - 1), Linear(width, J d)]``, last
            layer initialised to zero; skip: ``Linear(dz, J d)``; basis: the `CubicBSpline` of the knots.
    """

    def __init__(self, d: int, dz: int, knot_times: ArrayLike, width: int = 128, depth: int = 3, *,
                 seed: int = 0) -> None:
        super().__init__()
        d, dz, width, depth = int(d), int(dz), int(width), int(depth)
        if d < 1 or dz < 1:
            raise ValueError(f"d and dz must be >= 1, got d={d}, dz={dz}")
        if width < 1 or depth < 1:
            raise ValueError(f"width and depth must be >= 1, got width={width}, depth={depth}")
        self.d, self.dz, self.width, self.depth = d, dz, width, depth
        self.basis = CubicBSpline(knot_times)
        self.J: int = self.basis.J
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            f64 = {"dtype": torch.float64, "device": "cpu"}
            layers: list[nn.Module] = []
            w = dz
            for _ in range(depth):
                layers += [nn.Linear(w, width, **f64), nn.SiLU()]
                w = width
            layers.append(nn.Linear(w, self.J * d, **f64))
            self.trunk = nn.Sequential(*layers)
            self.skip = nn.Linear(dz, self.J * d, **f64)
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def extra_repr(self) -> str:
        return f"d={self.d}, dz={self.dz}, J={self.J}, width={self.width}, depth={self.depth}"

    def coeffs(self, z: Tensor) -> Tensor:
        """Control points ``(trunk(z) + skip(z))`` reshaped to (B, J, d) for latents z of shape (B, dz)."""
        return (self.trunk(z) + self.skip(z)).view(z.shape[0], self.J, self.d)

    def evaluate(self, t: float | ArrayLike | Tensor, z: Tensor, orders: Sequence[int] = (0,)) -> list[Tensor]:
        """Time derivatives of the sample paths at the normalised times ``t`` (Q values).

        ``out[i][q, b] = d^o/dt^o G(t_q, z_b)`` with ``o = orders[i]`` in {0, 1, 2}, computed as
        ``einsum("qj,bjd->qbd")`` of the basis derivative with the control points.

        Returns:
            One (Q, B, d) tensor per requested order, on the device of ``z``.
        """
        for o in orders:
            if o not in _ORDERS:
                raise ValueError(f"derivative orders must be in {_ORDERS}, got {o!r}")
        mats = self.basis(t, device=z.device)
        C = self.coeffs(z)                                   # (B, J, d)
        return [torch.einsum("qj,bjd->qbd", mats[o], C) for o in orders]

    def sample_latents(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        """Draw ``n`` latents ``z = randn(n, dz)`` from ``generator`` (on the model's device), shape (n, dz).

        The random stream of n particles is ``z`` followed by an (n, d) block of output noise that this model does not
        use (zero noise); the block is still drawn, which fixes the layout of the stream and so the particles of every
        seed (those of the paper's runs).
        """
        p = next(self.parameters())
        z = torch.randn(n, self.dz, generator=generator, device=p.device, dtype=p.dtype)
        torch.randn(n, self.d, generator=generator, device=p.device, dtype=p.dtype)
        return z

    @torch.no_grad()
    def init_from_data(self, frames: Sequence[np.ndarray]) -> None:
        """Initialise from the retained snapshots (one (n_k, d) array per knot time, in time order).

        With the pooled points P, their mean mu, ``Sigma = cov(P - mu) + 1e-8 I`` (unbiased) and ``L =
        cholesky(Sigma)`` (if Sigma is singular to working precision, a jitter ``1e-10 max(1, tr(Sigma) / d)`` is
        added instead, with a warning): the z = 0 control points C solve the K interpolation conditions
        ``sum_j B_j(t_k) C_j = mean(frame_k)`` and the natural end conditions ``G''(0) = G''(1) = 0`` in the least
        squares sense (written to the bias of the last trunk layer), and the skip weight of every control point is
        the first ``min(dz, d)`` columns of L (skip bias 0).
        """
        clouds = [np.asarray(c, dtype=np.float64) for c in frames]
        if len(clouds) != len(self.basis.knot_times):
            raise ValueError(f"{len(clouds)} snapshots for {len(self.basis.knot_times)} knot times")
        if any(c.ndim != 2 or c.shape[1] != self.d for c in clouds):
            raise ValueError(f"every snapshot must have shape (n, {self.d})")
        pooled = np.concatenate(clouds, 0)
        mu = pooled.mean(0)
        means = np.stack([c.mean(0) for c in clouds])
        d = pooled.shape[1]
        cov = np.atleast_2d(np.cov((pooled - mu).T)) + _COV_JITTER * np.eye(d)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            jitter = _COV_REL_JITTER * max(1.0, float(np.trace(cov)) / d)
            logger.warning("the pooled covariance of the snapshots is singular to working precision; adding %.3g "
                           "to its diagonal", jitter)
            L = np.linalg.cholesky(cov + jitter * np.eye(d))
        if self.dz <= d:
            L = L[:, :self.dz]
        Nn, _, D2 = basis_matrices(self.basis.knot_times, self.basis.knots)
        A_sys = np.vstack([Nn, D2[0][None, :], D2[-1][None, :]])
        b_sys = np.vstack([means, np.zeros((2, means.shape[1]))])
        C0, *_ = np.linalg.lstsq(A_sys, b_sys, rcond=None)        # (J, d)
        last = self.trunk[-1]
        last.bias.copy_(torch.as_tensor(C0.reshape(-1), dtype=last.bias.dtype))
        ncol = min(L.shape[1], self.dz)
        Wsk = np.zeros((self.J * self.d, self.dz))
        Wsk[:, :ncol] = np.tile(L[:, :ncol], (self.J, 1))        # rows j d : (j + 1) d = L[:, :ncol] for every j
        self.skip.weight.copy_(torch.as_tensor(Wsk, dtype=self.skip.weight.dtype))
        nn.init.zeros_(self.skip.bias)

    @torch.no_grad()
    def sample(self, tn: float, n: int, seed: int) -> Tensor:
        """``n`` samples of the marginal at NORMALISED time ``tn``, shape (n, d), on the model's device.

        All n latents are drawn first from ``torch.Generator(model device).manual_seed(seed)``, then the generator is
        evaluated in chunks of `SAMPLE_CHUNK`.
        """
        n = _positive_int(n, "number of samples")
        g = torch.Generator(device=next(self.parameters()).device).manual_seed(int(seed))
        z_all = self.sample_latents(n, generator=g)
        tq = [_scalar_time(tn)]
        outs = []
        for s in range(0, n, SAMPLE_CHUNK):
            X, = self.evaluate(tq, z_all[s:s + SAMPLE_CHUNK], orders=(0,))
            outs.append(X[0])
        return torch.cat(outs, 0)

    @torch.no_grad()
    def sample_paths(self, tn: ArrayLike, n: int, seed: int) -> Tensor:
        """Generator paths: the same ``n`` latents at every NORMALISED time of ``tn``, shape (Q, n, d).

        ``out[:, i]`` is the path of latent i; the latents are those of `sample` with the same seed.
        """
        n = _positive_int(n, "number of paths")
        tq = np.asarray(tn, dtype=np.float64).reshape(-1)
        if tq.size == 0:
            raise ValueError("times must contain at least one time")
        g = torch.Generator(device=next(self.parameters()).device).manual_seed(int(seed))
        z_all = self.sample_latents(n, generator=g)
        outs = []
        for s in range(0, n, SAMPLE_CHUNK):
            X, = self.evaluate(tq, z_all[s:s + SAMPLE_CHUNK], orders=(0,))
            outs.append(X)
        return torch.cat(outs, 1)

    def spec(self) -> dict[str, Any]:
        """Constructor arguments as plain python values (rebuild with `from_spec`, then load the state dict)."""
        return {"d": self.d, "dz": self.dz, "knot_times": [float(t) for t in self.basis.knot_times],
                "width": self.width, "depth": self.depth}

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> SplineGenerator:
        """A generator of the shapes of ``spec`` (seed 0); load the saved state dict afterwards."""
        return cls(spec["d"], spec["dz"], spec["knot_times"], spec["width"], spec["depth"], seed=0)
