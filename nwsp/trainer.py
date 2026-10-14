"""Training of NWSP and NWSP-N: configuration, training loop, potential learning, sampling, save and load.

The generator G (`nwsp.model`) maps a latent z to the control points of a clamped cubic B-spline over the normalised
retained times ``tn_k = (t_k - t_0) / (t_last - t_0)``. It is fitted to the retained snapshots mu_k by minimising

    L(theta) = 1/K sum_k [ S_eps(rho(tn_k), mu_k) + mmd_weight MMD^2(rho(tn_k), mu_k) ]
               + lambda(step) E_t E|a_M(t, X_t)|^2 / L^2,

with rho(t) the law of G(t, Z), S_eps the debiased Sinkhorn divergence and MMD^2 the multi-bandwidth Gaussian MMD on
minibatches of b_fit points per snapshot (`nwsp.losses`), a_M the projected acceleration (`nwsp.projection`) averaged
over n_times stratified times, and L the RMS distance of the pooled retained points from their mean (the penalty is
dimensionless). The weight is lambda(step) = 0 for step <= warmup_steps, then it rises linearly to lambda_acc over
max(1, int(ramp_frac (steps - warmup_steps))) steps; lambda_acc = 0 is the plain spline fit. Adam (learning rate lr,
cosine-annealed over all steps), gradient norm clipped at 10. S_eps is evaluated with sinkhorn_iters log-domain
iterations per potential and one differentiable refinement through both dual potentials: at converged potentials the
gradient of the Sinkhorn term is that of 2 S_eps (the logged value is S_eps); for a small eps (e.g. Petal, embryoid
body) 100 iterations do not fully converge, and the value and the gradient approximate these.

NWSP-N appends K learned potentials (`NeuralPotentialBlock`) to the fixed dictionary and, with the generator fixed,
trains them to capture as much of the population velocity as possible: with features F = [F_fixed | G / nrm] (G the
raw gradients of the learned potentials, nrm_k = sqrt(E|G_k|^2 + 1e-12) over both batches) and the moments
A_i = sym(E[F^T F]), b_i = E[F^T U] of two independent batches i = 1, 2 of the curve at a time t, a potential update
maximises the regularised explained velocity energy b^T (A + rI)^{-1} b normalised by the particle velocity energy
E|U|^2, cross-fitted on the two batches,

    Q(t) = 1/2 (b_2^T (A_1 + rI)^{-1} b_1 + b_1^T (A_2 + rI)^{-1} b_2) / (1/2 (E|U_1|^2 + E|U_2|^2)),

averaged over potential_times stratified times: potential_pretrain steps at the end of the warm-up, then one update
every potential_every generator steps, always before the generator step. The generator step treats the potentials as
constants.

One step: potential updates (NWSP-N), the generated and observed minibatches of every snapshot, the Sinkhorn
divergences of all snapshots, per snapshot ``fit = fit + (S_k + mmd_weight MMD_k)``, ``fit / K``, the penalty,
backward, clipping, Adam and scheduler steps. Terms that depend only on the observations (the y-y Sinkhorn potential
and MMD kernel) carry no gradient and are evaluated only at logging steps.

Random streams (the global torch / numpy generators are never used): generator initialisation ``seed``; dictionary and
MMD bandwidths ``dict_seed``, learned potentials ``dict_seed + 7000``; observation minibatches
``numpy.random.default_rng(seed + 1)``; latents and penalty times ``torch.Generator(device).manual_seed(seed + 2)``
(per step: the fit latents of every snapshot, then, while lambda > 0, the penalty times and per time the projection and
loss batches); potential updates ``seed + 3``.
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import math
import numbers
import re
import time
import typing
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .losses import auto_b_fit, auto_sinkhorn_eps, median_sigmas, rbf_mmd2, sinkhorn_divergence_frames, stratified_times
from .model import ArrayLike, SplineGenerator, _scalar_time
from .potentials import Dictionary, NeuralPotentialBlock, build_dictionary
from .projection import intrinsic_energy

logger = logging.getLogger(__name__)

GRAD_CLIP: float = 10.0
"""Maximum gradient norm of the generator (and of the learned potentials in a potential update)."""
LATENT_CAP: int = 16
"""``dz = "auto"`` gives ``min(d, LATENT_CAP)`` latent coordinates."""
NN_SEED_OFFSET: int = 7000
"""The learned potentials are initialised with the seed ``dict_seed + NN_SEED_OFFSET``."""
POTENTIAL_SEED_OFFSET: int = 3
"""The potential updates draw from a generator seeded ``seed + POTENTIAL_SEED_OFFSET``."""
SPREAD_RANGE: tuple[float, float] = (0.1, 20.0)
"""Per-coordinate RMS spread ``L / sqrt(d)`` of the retained snapshots for which the defaults are calibrated."""
REFRESH_TIMES: int = 4
"""Stratified times of a scale refresh of the learned potentials."""
REFRESH_BATCH: int = 256
"""Curve samples per refresh time."""
CHECKPOINT_VERSION: int = 1
"""Layout version of the files written by `NWSPTrainer.save`."""

_FIT_RANGE_TOL: float = 1e-9
_DEVICE_RE = re.compile(r"^(auto|cpu|cuda(:\d+)?)$")


# --------------------------------------------------------------------------------------------- configuration
@dataclass
class NWSPConfig:
    """Settings of an NWSP fit; the defaults are the paper's settings."""

    width: int = 128                 # generator MLP hidden width (SiLU)
    depth: int = 3                   # generator MLP hidden layers
    dz: int | str = "auto"           # latent dimension; "auto" = min(d, 16)
    n_ridge: int = 32                # requested ridge potentials; 6 * ceil(n_ridge / 6) are built (sine and cosine)
    ridge_reg: float = 1e-4          # ridge r of both projection solves
    lambda_acc: float = 1e-4         # penalty weight after warm-up and ramp; 0 = plain spline fit (no penalty)
    steps: int = 3000                # Adam steps
    warmup_steps: int = 300          # penalty off for step <= warmup_steps
    ramp_frac: float = 0.2           # linear ramp of the weight over ramp_frac * (steps - warmup_steps) steps
    lr: float = 3e-3                 # Adam learning rate, cosine-annealed to 0
    b_fit: int | str = "auto"        # points per snapshot in the fit term; "auto" = min(128, smallest snapshot)
    b_proj: int = 256                # projection batch of the penalty
    b_loss: int = 256                # loss batch of the penalty
    n_times: int = 2                 # stratified penalty times per step
    sinkhorn_eps: float | str = "auto"   # "auto" = max(1e-3, 0.02 * mean over snapshots of the summed variances)
    sinkhorn_iters: int = 100        # log-domain Sinkhorn iterations per potential
    mmd_weight: float = 0.1          # weight of the multi-bandwidth MMD^2
    seed: int = 0                    # training seed (generator init, minibatches, latents; potential updates: seed + 3)
    dict_seed: int = 0               # dictionary and MMD bandwidths (learned potentials: dict_seed + 7000)
    log_every: int = 100             # history record every log_every steps and at the last step
    device: str = "cpu"              # "cpu", "cuda", "cuda:N" or "auto" (CUDA when available and d > 10)

    def __post_init__(self) -> None:
        # python numbers (numpy scalars, integers given for float keys); invalid values are left for validate()
        floats = _float_fields(type(self))
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (bool, str)):
                continue
            if f.name in floats and isinstance(value, numbers.Real):
                setattr(self, f.name, float(value))
            elif f.name not in floats and isinstance(value, numbers.Integral):
                setattr(self, f.name, int(value))

    @classmethod
    def quickstart(cls, **overrides: Any) -> NWSPConfig:
        """Small networks and batches, 200 steps and eps fixed at 0.05, for the toy data; ``overrides`` win (the
        merging blobs of the ``converging`` toy data need ``steps=1000, warmup_steps=200``)."""
        return cls(**{**_QUICKSTART, **overrides})

    def validate(self) -> None:
        """Check every key; ``ValueError`` names the first offending key."""
        _check_int(self, "width", 1)
        _check_int(self, "depth", 1)
        _check_int_or_auto(self, "dz")
        _check_int(self, "n_ridge", 0)
        _check_float(self, "ridge_reg", 0.0, strict=True)
        _check_float(self, "lambda_acc", 0.0)
        _check_int(self, "steps", 1)
        _check_int(self, "warmup_steps", 0)
        if self.warmup_steps >= self.steps:
            raise ValueError(f"warmup_steps: {self.warmup_steps!r} must be smaller than steps ({self.steps})")
        _check_float(self, "ramp_frac", 0.0, strict=True)
        if self.ramp_frac > 1.0:
            raise ValueError(f"ramp_frac: {self.ramp_frac!r} must lie in (0, 1]")
        _check_float(self, "lr", 0.0, strict=True)
        _check_int_or_auto(self, "b_fit")
        _check_int(self, "b_proj", 1)
        _check_int(self, "b_loss", 1)
        _check_int(self, "n_times", 1)
        if self.sinkhorn_eps != "auto":
            _check_float(self, "sinkhorn_eps", 0.0, strict=True)
        _check_int(self, "sinkhorn_iters", 1)
        _check_float(self, "mmd_weight", 0.0)
        _check_int(self, "seed", 0)
        _check_int(self, "dict_seed", 0)
        _check_int(self, "log_every", 1)
        if not isinstance(self.device, str) or not _DEVICE_RE.match(self.device):
            raise ValueError(f"device: {self.device!r} is not 'auto', 'cpu', 'cuda' or 'cuda:N'")

    def to_dict(self) -> dict[str, Any]:
        """All keys as a plain dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> NWSPConfig:
        """Build from a mapping of keys; an unknown key raises ``ValueError``."""
        unknown = sorted(set(values) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise ValueError(f"unknown {cls.__name__} key(s): {', '.join(map(str, unknown))}")
        return cls(**dict(values))


@dataclass
class NWSPNConfig(NWSPConfig):
    """Settings of an NWSP-N fit: NWSPConfig plus the learned potentials and their training (the paper's design)."""

    n_neural: int = 16               # K learned potentials appended to the fixed dictionary
    nn_width: int = 64               # hidden width of the potentials' tanh MLP
    nn_depth: int = 2                # hidden layers (1 or 2)
    nn_gamma: float = 1.5            # scale of the first-layer initialisation
    potential_lr: float = 3e-3       # learning rate of the potential pretraining
    potential_lr_joint: float = 1e-3  # learning rate afterwards (cosine-annealed)
    potential_pretrain: int = 200    # potential pretraining steps at the end of the warm-up
    potential_every: int = 1         # then one potential update every potential_every generator steps
    potential_batch: int = 256       # samples per batch (two independent batches per time)
    potential_times: int = 2         # stratified times per potential update
    potential_refresh: int = 25      # feature scales refreshed every potential_refresh steps

    @classmethod
    def quickstart(cls, **overrides: Any) -> NWSPNConfig:
        """`NWSPConfig.quickstart` with 8 learned potentials of width 32, 20 potential pretraining steps and potential
        batches of 64."""
        return super().quickstart(**{**_QUICKSTART_NEURAL, **overrides})

    def validate(self) -> None:
        """Check every key; ``ValueError`` names the first offending key."""
        super().validate()
        _check_int(self, "n_neural", 1)
        _check_int(self, "nn_width", 1)
        _check_int(self, "nn_depth", 1)
        if self.nn_depth not in (1, 2):
            raise ValueError(f"nn_depth: {self.nn_depth!r} is not 1 or 2")
        _check_float(self, "nn_gamma", 0.0, strict=True)
        _check_float(self, "potential_lr", 0.0, strict=True)
        _check_float(self, "potential_lr_joint", 0.0)
        _check_int(self, "potential_pretrain", 0)
        _check_int(self, "potential_every", 1)
        _check_int(self, "potential_batch", 1)
        _check_int(self, "potential_times", 1)
        _check_int(self, "potential_refresh", 1)


_QUICKSTART: dict[str, Any] = {"width": 48, "depth": 2, "n_ridge": 12, "lambda_acc": 1e-3, "steps": 200,
                               "warmup_steps": 40, "b_fit": 64, "b_proj": 64, "b_loss": 64, "sinkhorn_eps": 0.05,
                               "log_every": 10}
_QUICKSTART_NEURAL: dict[str, Any] = {"n_neural": 8, "nn_width": 32, "potential_pretrain": 20, "potential_batch": 64}


@functools.cache
def _float_fields(cls: type) -> frozenset[str]:
    """Names of the fields whose declared type admits a float."""
    hints = typing.get_type_hints(cls)
    return frozenset(name for name, tp in hints.items() if tp is float or float in typing.get_args(tp))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_int(cfg: Any, key: str, minimum: int) -> None:
    value = getattr(cfg, key)
    if not _is_int(value) or value < minimum:
        raise ValueError(f"{key}: {value!r} is not an integer >= {minimum}")


def _check_int_or_auto(cfg: Any, key: str) -> None:
    value = getattr(cfg, key)
    if value != "auto" and (not _is_int(value) or value < 1):
        raise ValueError(f"{key}: {value!r} is not 'auto' or an integer >= 1")


def _check_float(cfg: Any, key: str, minimum: float, *, strict: bool = False) -> None:
    value = getattr(cfg, key)
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not ok or value < minimum or (strict and value == minimum):
        raise ValueError(f"{key}: {value!r} is not a finite number {'>' if strict else '>='} {minimum:g}")


def _resolve_config(config: NWSPConfig, frames: Sequence[np.ndarray]) -> NWSPConfig:
    """Copy of ``config`` with ``dz``, ``b_fit``, ``sinkhorn_eps`` and ``device`` resolved from the retained frames."""
    d = int(frames[0].shape[1])
    dz = min(d, LATENT_CAP) if config.dz == "auto" else config.dz
    b_fit = auto_b_fit(frames) if config.b_fit == "auto" else config.b_fit
    eps = auto_sinkhorn_eps(frames) if config.sinkhorn_eps == "auto" else config.sinkhorn_eps
    device = config.device
    if device == "auto":
        device = "cuda" if (torch.cuda.is_available() and d > 10) else "cpu"
    return dataclasses.replace(config, dz=dz, b_fit=int(b_fit), sinkhorn_eps=float(eps), device=device)


def _torch_device(name: str) -> torch.device:
    """``torch.device(name)``; ``ValueError`` when it names a CUDA device that this process cannot use."""
    dev = torch.device(name)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"device: {name!r} requested but CUDA is not available (use 'cpu' or 'auto')")
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise ValueError(f"device: {name!r} requested but only {torch.cuda.device_count()} GPU(s) are visible")
    return dev


def _check_snapshots(times: ArrayLike, frames: Sequence[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Validated float64 copies: times (T,) finite and strictly increasing, T >= 2; T finite (n_t >= 1, d) frames."""
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError(f"times must be a 1-D sequence of at least two times, got shape {t.shape}")
    if not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("times must be finite and strictly increasing")
    clouds = [np.array(c, dtype=np.float64, order="C") for c in frames]
    if len(clouds) != len(t):
        raise ValueError(f"got {len(t)} times but {len(clouds)} snapshots")
    for k, c in enumerate(clouds):
        if c.ndim != 2 or c.shape[0] < 1 or c.shape[1] < 1:
            raise ValueError(f"snapshot {k} must be an (n, d) array with n >= 1, got shape {c.shape}")
        if c.shape[1] != clouds[0].shape[1]:
            raise ValueError(f"snapshot {k} has dimension {c.shape[1]}, snapshot 0 has {clouds[0].shape[1]}")
        if not np.all(np.isfinite(c)):
            raise ValueError(f"snapshot {k} contains non-finite values")
    return t, clouds


def _normalize_times(times: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Affine map of the retained times onto [0, 1]: ``((t - t_0) / (t_last - t_0), t_0, t_last - t_0)``."""
    t0, t1 = float(times[0]), float(times[-1])
    return (times - t0) / (t1 - t0), t0, t1 - t0


def _warn_outside(tn: np.ndarray, t: np.ndarray, t_origin: float, t_scale: float) -> None:
    """``UserWarning`` for the physical times ``t`` whose normalised times ``tn`` lie outside [0, 1]."""
    outside = t[(tn < -_FIT_RANGE_TOL) | (tn > 1.0 + _FIT_RANGE_TOL)]
    if outside.size:
        shown = ", ".join(f"{x:g}" for x in outside[:5]) + (", ..." if outside.size > 5 else "")
        warnings.warn(f"time(s) {shown} outside the fitted range [{t_origin:g}, {t_origin + t_scale:g}]: the spline "
                      f"is extrapolated with the cubic of its first / last span", UserWarning, stacklevel=3)


@contextlib.contextmanager
def _frozen(module: nn.Module | None) -> Iterator[None]:
    """The parameters of ``module`` do not require gradients inside the block (restored afterwards)."""
    if module is None:
        yield
        return
    params = list(module.parameters())
    previous = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in zip(params, previous):
            p.requires_grad_(flag)


# ---------------------------------------------------------------------------------------- potential learning
@torch.no_grad()
def _sample_curve(gen: SplineGenerator, t: float, n: int, generator: torch.Generator,
                  orders: Sequence[int] = (0, 1)) -> list[Tensor]:
    """Detached samples of the curve at normalised time t: ``[X_t, U_t]`` (or the requested orders), each (n, d)."""
    z = gen.sample_latents(n, generator=generator)
    return [o[0] for o in gen.evaluate([float(t)], z, orders=orders)]


def _gram(F: Tensor, U: Tensor) -> tuple[Tensor, Tensor]:
    """Moments of one batch: ``A = sym(E[F^T F])`` (M, M) and ``b = E[F^T U]`` (M,)."""
    B = F.shape[0]
    A = torch.einsum("bdm,bdn->mn", F, F) / B
    b = torch.einsum("bdm,bd->m", F, U) / B
    return 0.5 * (A + A.transpose(0, 1)), b


def _quad(A: Tensor, b1: Tensor, b2: Tensor, ridge: float) -> Tensor:
    """``b2^T (A + rI)^{-1} b1`` through one Cholesky factorisation."""
    H = A + ridge * torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    L = torch.linalg.cholesky(H)
    return (b2 * torch.cholesky_solve(b1.unsqueeze(1), L).squeeze(1)).sum()


def _feature_norm(grads: Sequence[Tensor]) -> Tensor:
    """Differentiable normaliser ``sqrt(E|G_k|^2 + 1e-12)`` of each learned feature over the given batches, (K,)."""
    return torch.cat(list(grads), 0).pow(2).sum(1).mean(0).add(1e-12).sqrt()


def _explained_energy_loss(block: NeuralPotentialBlock, fixed_dic: Dictionary, gen: SplineGenerator, times: Tensor,
                           batch: int, ridge: float, generator: torch.Generator) -> tuple[Tensor, float]:
    """Loss ``-mean_t Q(t)`` of a potential update (differentiable in the block) and the mean score Q.

    Per time: two independent detached batches of ``batch`` curve samples (batch 1, then batch 2), the fixed features
    without gradient, the learned features normalised by `_feature_norm`, and the cross-fitted normalised explained
    velocity energy Q(t) of the module docstring (a stationary curve gives 0).
    """
    Q_tot: Any = 0.0
    for t in times:
        X1, U1 = _sample_curve(gen, float(t), batch, generator)
        X2, U2 = _sample_curve(gen, float(t), batch, generator)
        with torch.no_grad():
            Ff1, Ff2 = fixed_dic.grad(X1), fixed_dic.grad(X2)
        Gr1, Gr2 = block.grad_raw(X1), block.grad_raw(X2)
        nrm = _feature_norm([Gr1, Gr2])
        A1, b1 = _gram(torch.cat([Ff1, Gr1 / nrm], 2), U1)
        A2, b2 = _gram(torch.cat([Ff2, Gr2 / nrm], 2), U2)
        tiny = torch.finfo(U1.dtype).tiny
        e_u = (0.5 * ((U1 ** 2).sum(1).mean() + (U2 ** 2).sum(1).mean())).clamp_min(tiny)
        Q = 0.5 * (_quad(A1, b1, b2, ridge) + _quad(A2, b2, b1, ridge)) / e_u
        Q_tot = Q_tot + Q
    n = len(times)
    return -(Q_tot / n), float(Q_tot.detach()) / n


class _PotentialLearner:
    """Schedule of the potential updates inside an NWSP-N fit; `update` runs before generator step ``step``.

    At ``step == warmup_steps`` the block is pretrained for ``potential_pretrain`` steps at ``potential_lr`` (scale
    refresh at pretraining steps ``i % potential_refresh == 0`` and once at the end); then the learning rate is set to
    ``potential_lr_joint`` and annealed by ``CosineAnnealingLR(T_max = max(1, steps - warmup_steps))``. For ``step >
    warmup_steps`` with ``(step - warmup_steps) % potential_every == 0``: one potential update (scale refresh when
    ``step % potential_refresh == 0``) followed by one scheduler step. A potential update: stratified times,
    `_explained_energy_loss`, backward, gradient norm clipped at 10, Adam step. Random stream: seed + 3 on the model's
    device.
    """

    def __init__(self, block: NeuralPotentialBlock, fixed_dic: Dictionary, gen: SplineGenerator,
                 config: NWSPNConfig) -> None:
        self.block, self.fixed_dic, self.gen, self.config = block, fixed_dic, gen, config
        self.device = next(gen.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed + POTENTIAL_SEED_OFFSET)
        self.optimizer = torch.optim.Adam(block.parameters(), lr=config.potential_lr)
        self.scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None   # set by _pretrain

    def update(self, step: int) -> float | None:
        """Potential updates scheduled before generator step ``step``: Q of the last update run, or None."""
        cfg = self.config
        if step == cfg.warmup_steps:
            return self._pretrain()
        if step > cfg.warmup_steps and (step - cfg.warmup_steps) % cfg.potential_every == 0:
            Q = self._step(refresh=step % cfg.potential_refresh == 0)
            self.scheduler.step()
            return Q
        return None

    def refresh_scale(self) -> None:
        """Rescale the learned potentials so that E|grad g_k|^2 = 1 on REFRESH_TIMES x REFRESH_BATCH curve samples."""
        ts = stratified_times(REFRESH_TIMES, generator=self.generator, device=self.device)
        X = torch.cat([_sample_curve(self.gen, float(t), REFRESH_BATCH, self.generator, orders=(0,))[0]
                       for t in ts], 0)
        self.block.refresh_scale(X)

    def _step(self, refresh: bool) -> float:
        cfg = self.config
        ts = stratified_times(cfg.potential_times, generator=self.generator, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        loss, Q = _explained_energy_loss(self.block, self.fixed_dic, self.gen, ts, cfg.potential_batch, cfg.ridge_reg,
                                         self.generator)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.block.parameters(), GRAD_CLIP)
        self.optimizer.step()
        if refresh:
            self.refresh_scale()
        return Q

    def _pretrain(self) -> float | None:
        cfg = self.config
        Q = None
        for i in range(cfg.potential_pretrain):
            Q = self._step(refresh=i % cfg.potential_refresh == 0)
        self.refresh_scale()
        for group in self.optimizer.param_groups:
            group["lr"] = cfg.potential_lr_joint
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, cfg.steps - cfg.warmup_steps))
        if Q is not None:
            logger.info("potential pretraining (%d steps): Q = %.4f", cfg.potential_pretrain, Q)
        return Q


def _make_neural_block(config: NWSPNConfig, pooled: np.ndarray, pooled_mean: np.ndarray, length_scale: float,
                       device: torch.device) -> NeuralPotentialBlock:
    """The learned potentials of NWSP-N: centred and scaled by the pooled retained points (mean, L), with their
    feature scales set on these points."""
    block = NeuralPotentialBlock(pooled.shape[1], config.n_neural, width=config.nn_width, depth=config.nn_depth,
                                 center=pooled_mean, length=length_scale, seed=config.dict_seed + NN_SEED_OFFSET,
                                 init_gamma=config.nn_gamma).to(device)
    block.refresh_scale(torch.as_tensor(pooled, dtype=torch.float64, device=device))
    return block


# --------------------------------------------------------------------------------------------------- trainers
class NWSPTrainer:
    """Fits NWSP to snapshots; after `fit`, samples the fitted curve at any time.

    Args:
        times: (T,) physical snapshot times, strictly increasing, T >= 2.
        frames: T arrays (n_t, d) of finite points (the snapshots to fit).
        config: an `NWSPConfig` (default: the paper's settings).

    Attributes:
        config: the resolved configuration (``dz``, ``b_fit``, ``sinkhorn_eps``, ``device`` filled in);
        device; times (physical), tn (normalised), t_origin, t_scale (``tn = (t - t_origin) / t_scale``);
        frames (None for a loaded model); d; gen (`SplineGenerator`); dic_fixed (fixed dictionary); dic (the
        dictionary of the penalty: dic_fixed, plus the learned block for NWSP-N); nn_block; sigmas (MMD bandwidths);
        length_scale (L) and acc_scale (L^2); history (one record per logging step: step, loss, fit, sinkhorn, mmd,
        lambda_acc, sec, [acc_energy], [potential_Q]); train_seconds; whitening ((mean, std) of the training
        coordinates, set by `save` and `load_model`).
    """

    method: str = "NWSP"

    def __init__(self, times: ArrayLike, frames: Sequence[np.ndarray], config: NWSPConfig | None = None) -> None:
        if config is None:
            config = NWSPConfig()
        if isinstance(config, NWSPNConfig) and not isinstance(self, NWSPNTrainer):
            raise TypeError("an NWSPNConfig describes an NWSP-N fit: use NWSPNTrainer")
        if not isinstance(config, NWSPConfig):
            raise TypeError(f"config must be an NWSPConfig, got {type(config).__name__}")
        times, frames = _check_snapshots(times, frames)
        config.validate()
        config = _resolve_config(config, frames)
        config.validate()
        device = _torch_device(config.device)
        d = int(frames[0].shape[1])

        gen = SplineGenerator(d, config.dz, _normalize_times(times)[0], config.width, config.depth,
                              seed=config.seed).to(device)
        gen.init_from_data(frames)
        pooled = np.concatenate(frames, 0)
        pooled_mean = pooled.mean(0)
        dic_fixed = build_dictionary(pooled, n_ridge=config.n_ridge, seed=config.dict_seed, device=device)
        sigmas = median_sigmas(pooled, seed=config.dict_seed)
        length_scale = max(float(np.sqrt(np.mean(((pooled - pooled_mean) ** 2).sum(1)))), 1e-12)
        spread = length_scale / math.sqrt(d)
        if not SPREAD_RANGE[0] <= spread <= SPREAD_RANGE[1]:
            logger.warning("per-coordinate RMS spread %.3g of the snapshots is outside [%g, %g]: coordinates far from "
                           "unit scale: standardise the snapshots (data_gen.standardize) and map samples back with "
                           "x * std + mean", spread, *SPREAD_RANGE)
        nn_block = (_make_neural_block(config, pooled, pooled_mean, length_scale, device)
                    if isinstance(config, NWSPNConfig) else None)
        self._set_model(config, device, times, gen, dic_fixed, nn_block, sigmas, length_scale)
        self.frames = frames
        # the snapshots live on the device once; each step gathers its minibatches by index
        self._obs = [torch.as_tensor(c, dtype=torch.float64, device=device) for c in frames]

        n_neural = 0 if nn_block is None else nn_block.K
        logger.info("%s: d=%d, %d retained snapshots (sizes %s), dz=%d, %d fixed potentials%s, b_fit=%d, eps=%.4g, "
                    "L=%.4g, device=%s", self.method, d, len(frames), [len(c) for c in frames], config.dz,
                    dic_fixed.M, f" + {n_neural} learned" if n_neural else "", config.b_fit,
                    config.sinkhorn_eps, length_scale, device)

    def _set_model(self, config: NWSPConfig, device: torch.device, times: np.ndarray, gen: SplineGenerator,
                   dic_fixed: Dictionary, nn_block: NeuralPotentialBlock | None, sigmas: Sequence[float],
                   length_scale: float) -> None:
        """Set the attributes of a model (a new trainer and `load_model`); no snapshots, history or whitening yet."""
        self.config, self.device, self.d = config, device, gen.d
        self.times = times
        self.tn, self.t_origin, self.t_scale = _normalize_times(times)
        self.gen, self.dic_fixed, self.nn_block = gen, dic_fixed, nn_block
        self.dic = dic_fixed if nn_block is None else Dictionary([*dic_fixed.blocks, nn_block])
        self.sigmas = tuple(float(s) for s in sigmas)
        self.length_scale = float(length_scale)
        self.acc_scale = self.length_scale ** 2
        self.frames: list[np.ndarray] | None = None
        self._obs: list[Tensor] = []
        self.history: list[dict[str, Any]] = []
        self.train_seconds: float | None = None
        self.whitening: tuple[np.ndarray, np.ndarray] | None = None
        self._fitted = False

    def _make_potential_learner(self) -> _PotentialLearner | None:
        return None

    def penalty_weight(self, step: int) -> float:
        """lambda(step): 0 for ``step <= warmup_steps``, then ``min(1, (step - warmup_steps) / ramp) lambda_acc``
        with ``ramp = max(1, int(ramp_frac (steps - warmup_steps)))``."""
        cfg = self.config
        if step < cfg.warmup_steps:
            return 0.0
        ramp = max(1, int(cfg.ramp_frac * (cfg.steps - cfg.warmup_steps)))
        return min(1.0, (step - cfg.warmup_steps) / ramp) * cfg.lambda_acc

    def fit(self) -> NWSPTrainer:
        """Run ``config.steps`` training steps; returns the trainer.

        Raises:
            FloatingPointError: the loss of a logging step is not finite (the fit diverged).
            torch.linalg.LinAlgError: a projection or potential-learning system cannot be factorised (the fit diverged).
            RuntimeError: ``fit`` was already called, or the model was loaded from a checkpoint.
        """
        if self._fitted:
            raise RuntimeError("this trainer is already fitted (fit() runs once per trainer)")
        self._fitted = True
        cfg, device, gen = self.config, self.device, self.gen
        opt = torch.optim.Adam(gen.parameters(), lr=cfg.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
        rng = np.random.default_rng(cfg.seed + 1)
        g = torch.Generator(device=device).manual_seed(cfg.seed + 2)
        learner = self._make_potential_learner()
        batched = device.type == "cpu"     # Sinkhorn potentials of all snapshots at once on the CPU, one by one on CUDA
        sizes = [len(c) for c in self._obs]
        n_obs = [min(cfg.b_fit, n) for n in sizes]
        K = len(sizes)
        potential_Q = None

        t_start = time.perf_counter()
        try:
            for step in range(cfg.steps):
                opt.zero_grad(set_to_none=True)
                lam = self.penalty_weight(step)
                log = step % cfg.log_every == 0 or step == cfg.steps - 1
                if learner is not None:
                    Q = learner.update(step)
                    if Q is not None:
                        potential_Q = Q

                # fit term: generated points and observation minibatches of every snapshot, then the losses
                xs = []
                for k in range(K):
                    z = gen.sample_latents(cfg.b_fit, generator=g)
                    xs.append(gen.evaluate([self.tn[k]], z)[0][0])
                idx = [rng.choice(sizes[k], size=n_obs[k], replace=False) for k in range(K)]
                idx = torch.as_tensor(np.concatenate(idx), device=device).split(n_obs)
                ys = [self._obs[k][idx[k]] for k in range(K)]
                sink = sinkhorn_divergence_frames(xs, ys, cfg.sinkhorn_eps, cfg.sinkhorn_iters, with_constant=log,
                                                  batched=batched)
                mmd = []
                fit_total: Any = 0.0
                for k in range(K):
                    m_k = rbf_mmd2(xs[k], ys[k], self.sigmas, with_constant=log)
                    fit_total = fit_total + (sink[k] + cfg.mmd_weight * m_k)
                    mmd.append(m_k)
                fit_total = fit_total / K

                # penalty: no draws at all while its weight is zero
                pen = torch.zeros((), dtype=torch.float64, device=device)
                acc = None
                if lam > 0:
                    ts = stratified_times(cfg.n_times, generator=g, device=device).tolist()
                    with _frozen(self.nn_block):
                        acc = torch.stack([intrinsic_energy(gen, self.dic, t, cfg.b_proj, cfg.b_loss, cfg.ridge_reg, g)
                                           for t in ts]).mean()
                        pen = (lam * acc) / self.acc_scale
                        loss = fit_total + pen
                        loss.backward()
                else:
                    loss = fit_total + pen
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
                opt.step()
                sched.step()

                if log:
                    rec = _record(step, loss, fit_total, sink, mmd, lam, acc, potential_Q,
                                  time.perf_counter() - t_start)
                    self.history.append(rec)
                    logger.info(_format_record(rec, cfg.steps))
                    if not math.isfinite(rec["loss"]):
                        raise FloatingPointError(f"the training loss is not finite at step {step} (the fit diverged)")
        finally:
            self.train_seconds = time.perf_counter() - t_start
        logger.info("%s: %d steps in %.1f s", self.method, cfg.steps, self.train_seconds)
        return self

    # ------------------------------------------------------------------------------------------------ sampling
    def sample(self, t: float, n: int, seed: int = 0) -> np.ndarray:
        """``n`` samples of the fitted marginal at PHYSICAL time ``t``, shape (n, d), in training coordinates.

        One generator evaluation; latents from a generator seeded ``seed`` on the model's device (CPU and CUDA give
        different streams). Times outside [times[0], times[-1]] extrapolate the spline (``UserWarning``).
        """
        t = _scalar_time(t)
        tn = (t - self.t_origin) / self.t_scale
        _warn_outside(np.array([tn]), np.array([t]), self.t_origin, self.t_scale)
        return self.gen.sample(tn, n, seed).cpu().numpy()

    def sample_paths(self, times: ArrayLike, n: int, seed: int = 0) -> np.ndarray:
        """Generator paths: the same ``n`` particles at every PHYSICAL time of ``times``, shape (Q, n, d).

        ``out[:, i]`` is the path t -> G(t, z_i) of latent i (the latents of `sample` with the same seed): a path of
        the sampling representation, not an identified cell trajectory.
        """
        t = np.asarray(times, dtype=np.float64).reshape(-1)
        tn = (t - self.t_origin) / self.t_scale
        _warn_outside(tn, t, self.t_origin, self.t_scale)
        return self.gen.sample_paths(tn, n, seed).cpu().numpy()

    # ------------------------------------------------------------------------------------------- save and load
    def save(self, path: str | Path, whitening: tuple[np.ndarray, np.ndarray] | None = None) -> Path:
        """Write a weights-only checkpoint (CPU tensors and plain containers; read it back with `load_model`).

        ``whitening`` = (mean, std) of the standardisation of the training coordinates (data = x * std + mean); it is
        stored with the model and on the trainer. Parent directories are created. Returns the path.
        """
        path = Path(path)
        if whitening is not None:
            mean, std = (np.asarray(v, dtype=np.float64).reshape(-1) for v in whitening)
            if mean.shape != (self.d,) or std.shape != (self.d,):
                raise ValueError(f"whitening mean and std must have shape ({self.d},), got {mean.shape} and "
                                 f"{std.shape}")
            self.whitening = (mean, std)
        w = self.whitening
        block = self.nn_block
        state = {
            "nwsp_checkpoint": CHECKPOINT_VERSION,
            "method": self.method,
            "config": self.config.to_dict(),
            "generator": self.gen.spec(),
            "generator_state": _cpu_state(self.gen),
            "dictionary": self.dic_fixed.spec(),
            "dictionary_state": _cpu_state(self.dic_fixed),
            "neural": None if block is None else block.spec(),
            "neural_state": None if block is None else _cpu_state(block),
            "times": [float(t) for t in self.times],
            "t_origin": float(self.t_origin),
            "t_scale": float(self.t_scale),
            "length_scale": float(self.length_scale),
            "sigmas": [float(s) for s in self.sigmas],
            "whitening": None if w is None else {"mean": [float(v) for v in w[0]], "std": [float(v) for v in w[1]]},
            "history": [dict(rec) for rec in self.history],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        return path


class NWSPNTrainer(NWSPTrainer):
    """NWSP-N: NWSP with K learned potentials trained by the potential-learning objective.

    Args:
        times, frames: as in `NWSPTrainer`.
        config: an `NWSPNConfig` (default: the paper's NWSP-N settings).
    """

    method: str = "NWSP-N"

    def __init__(self, times: ArrayLike, frames: Sequence[np.ndarray], config: NWSPNConfig | None = None) -> None:
        if config is None:
            config = NWSPNConfig()
        if not isinstance(config, NWSPNConfig):
            raise TypeError(f"NWSPNTrainer needs an NWSPNConfig, got {type(config).__name__} (use NWSPTrainer)")
        super().__init__(times, frames, config)
        if config.lambda_acc == 0:
            logger.info("lambda_acc = 0: the learned potentials are unused and the fit equals NWSP")

    def _make_potential_learner(self) -> _PotentialLearner | None:
        if self.nn_block is None or self.config.lambda_acc <= 0:
            return None
        return _PotentialLearner(self.nn_block, self.dic_fixed, self.gen, self.config)


def load_model(path: str | Path, device: str | torch.device | None = None) -> NWSPTrainer:
    """Rebuild a fitted model from a checkpoint written by `NWSPTrainer.save`.

    Returns an `NWSPTrainer` (an `NWSPNTrainer` for NWSP-N) ready for `sample` / `sample_paths`; it cannot be
    trained further. ``device``: None (default) = the training device when available here, else the CPU (with a
    warning: CPU and CUDA random streams differ, so CPU samples of a GPU-trained model differ from its GPU samples).
    """
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or ck.get("nwsp_checkpoint") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} is not an NWSP checkpoint (version {CHECKPOINT_VERSION})")
    neural = ck["method"] == "NWSP-N"
    config = (NWSPNConfig if neural else NWSPConfig).from_dict(ck["config"])
    if device is not None:
        dev = _torch_device(str(device))
    else:
        dev = torch.device(config.device)
        if dev.type == "cuda" and not (torch.cuda.is_available()
                                       and (dev.index is None or dev.index < torch.cuda.device_count())):
            logger.warning("%s was trained on %s, which is not available: loading it on the CPU (CPU and CUDA random "
                           "streams differ, so the samples differ from those of the GPU)", path, config.device)
            dev = torch.device("cpu")

    gen = SplineGenerator.from_spec(ck["generator"])
    gen.load_state_dict(ck["generator_state"], strict=True)
    fixed = Dictionary.from_spec(ck["dictionary"])
    fixed.load_state_dict(ck["dictionary_state"], strict=True)
    block = None
    if ck["neural"] is not None:
        block = NeuralPotentialBlock.from_spec(ck["neural"])
        block.load_state_dict(ck["neural_state"], strict=True)
        block.to(dev).eval()
    gen.to(dev).eval()
    fixed.to(dev).eval()

    cls = NWSPNTrainer if neural else NWSPTrainer
    model = cls.__new__(cls)
    model._set_model(config, dev, np.asarray(ck["times"], dtype=np.float64), gen, fixed, block, ck["sigmas"],
                     ck["length_scale"])
    model.history = list(ck["history"])
    w = ck["whitening"]
    if w is not None:
        model.whitening = (np.asarray(w["mean"], dtype=np.float64), np.asarray(w["std"], dtype=np.float64))
    model._fitted = True      # a loaded model samples; it is not trained further
    return model


# ----------------------------------------------------------------------------------------------------- helpers
def _cpu_state(module: nn.Module) -> dict[str, Tensor]:
    """A state dict as a plain dict of detached CPU copies."""
    return {str(k): v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


def _record(step: int, loss: Tensor, fit_total: Tensor, sink: Sequence[Tensor], mmd: Sequence[Tensor], lam: float,
            acc: Tensor | None, potential_Q: float | None, seconds: float) -> dict[str, Any]:
    """History record of a logging step (python floats; the per-snapshot terms summed in snapshot order)."""
    s_val = 0.0
    for s in sink:
        s_val += s.item()
    m_val = 0.0
    for m in mmd:
        m_val += m.item()
    n = len(sink)
    rec: dict[str, Any] = {"step": step, "loss": loss.item(), "fit": fit_total.item(), "sinkhorn": s_val / n,
                           "mmd": m_val / n, "lambda_acc": lam, "sec": round(seconds, 1)}
    if acc is not None:
        rec["acc_energy"] = acc.item()
    if potential_Q is not None:
        rec["potential_Q"] = potential_Q
    return rec


def _format_record(rec: dict[str, Any], steps: int) -> str:
    parts = [f"step {rec['step']:>{len(str(steps))}d}/{steps}", f"loss {rec['loss']:.6g}", f"fit {rec['fit']:.6g}",
             f"lambda {rec['lambda_acc']:.3g}"]
    if "acc_energy" in rec:
        parts.append(f"acc {rec['acc_energy']:.4g}")
    if "potential_Q" in rec:
        parts.append(f"potential Q {rec['potential_Q']:.4f}")
    parts.append(f"{rec['sec']:.1f} s")
    return "  ".join(parts)
