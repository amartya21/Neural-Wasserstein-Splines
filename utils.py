"""
Evaluation and plotting helpers:
 - evaluate : exact W1 and W2 (POT network simplex) and the multi-bandwidth MMD between samples and a snapshot
 - evaluate_split : the paper's scores of a fitted model at the held-out (and retained) snapshots of a split
 - plot_figure : generator paths coloured by time, with the observed snapshots on top
 - plot_snapshots : observed (top row) and generated (bottom row) snapshots at the observed times
 - save_json : JSON output (numpy values converted, NaN and inf written as null)

Scores of samples x_1..x_n (weights a_i = 1/n) against a snapshot y_1..y_m with weights b_j:
 - W1 = min_P sum_ij P_ij |x_i - y_j| over the couplings P of (a, b);
 - W2 = (min_P sum_ij P_ij |x_i - y_j|^2)^(1/2);
 - MMD = (1/5) sum_s [a^T K_s(X, X) a + b^T K_s(Y, Y) b - 2 a^T K_s(X, Y) b], the squared MMD reported in the paper,
   with K_s(x, y) = exp(-|x - y|^2 / (2 s^2)) and s = (0.25, 0.5, 1, 2, 4) x the standard deviation of all
   coordinates of the snapshot.
The couplings are exact network-simplex solutions (POT ot.lp.emd, one thread). Memory is O(n m) for W1 and W2 and
O((n + m)^2) for the MMD.
"""
from __future__ import annotations

import json
import logging
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ot
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from data_gen import Split

logger = logging.getLogger(__name__)

HELD_OUT_SEED_OFFSET: int = 1000   # samples scored at a held-out snapshot: seed 1000 + training seed
RETAINED_SEED_OFFSET: int = 2000   # samples of the retained-snapshot W1 diagnostic: seed 2000 + training seed
METRICS: tuple[str, ...] = ("W1", "W2", "MMD")
MMD_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
EMD_ITERMAX: int = 10_000_000      # network-simplex iterations; one retry with 20 times as many
PLOT_MAX_POINTS: int = 2000        # points drawn per snapshot (a fixed random subset of larger ones)
BLUE, RED = "#2F6FD0", "#C62828"


# ---------------------------------------------------------------------------------------------------- scores
def evaluate(samples: np.ndarray, truth: np.ndarray, truth_weights: np.ndarray | None = None,
             metrics: Sequence[str] = METRICS) -> dict[str, float]:
    """Scores of `samples` (n, d), a uniform measure, against the snapshot `truth` (m, d).

    Args:
        samples: generated points.
        truth: observed points.
        truth_weights: (m,) weights of the observed points (default uniform); normalised to sum 1.
        metrics: any of "W1", "W2", "MMD".

    Returns:
        {metric: value} as Python floats; every value is NaN when a sample is not finite (no solver call).
    """
    metrics = tuple(metrics)
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; expected a subset of {METRICS}")
    X, Y = _points(samples), _points(truth)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dimension mismatch: samples have d = {X.shape[1]}, the snapshot d = {Y.shape[1]}")
    if not np.all(np.isfinite(X)):
        return dict.fromkeys(metrics, math.nan)
    a = _probabilities(np.full(len(X), 1.0 / len(X)))
    b = _probabilities(np.full(len(Y), 1.0 / len(Y)) if truth_weights is None else truth_weights)
    if len(b) != len(Y):
        raise ValueError(f"{len(b)} weights for {len(Y)} points")
    out: dict[str, float] = {}
    if "W1" in metrics:
        C = cdist(X, Y, "euclidean")
        out["W1"] = float((_emd(a, b, C) * C).sum())
    if "W2" in metrics:
        C = cdist(X, Y, "sqeuclidean")
        value = float((_emd(a, b, C) * C).sum())
        out["W2"] = math.nan if math.isnan(value) else float(np.sqrt(max(0.0, value)))
    if "MMD" in metrics:
        out["MMD"] = _mmd(X, Y, a, b)
    return {m: out[m] for m in metrics}


def evaluate_split(trainer: Any, split: Split, retained: bool = True) -> dict[str, dict]:
    """The paper's scores of a model fitted on `split`.

    Every held-out snapshot k: n_k = |snapshot k| samples drawn at time t_k with seed 1000 + training seed, scored
    with `evaluate` (W1, W2, MMD). With `retained`, also the W1 at every retained snapshot i (a fit diagnostic;
    samples drawn with seed 2000 + training seed).

    Args:
        trainer: a fitted NWSPTrainer / NWSPNTrainer or a loaded model (uses trainer.sample(t, n, seed) and
            trainer.config.seed).
        split: the data_gen.Split the model was fitted on (scores are in its training coordinates).
        retained: also score the retained snapshots.

    Returns:
        {"held_out": {k: {"W1", "W2", "MMD"}}, "retained_w1": {i: W1}, "samples": {k or i: (n, d) samples}}.
    """
    seed = int(trainer.config.seed)
    out: dict[str, dict] = {"held_out": {}, "retained_w1": {}, "samples": {}}
    for k in split.held_out:
        X = np.asarray(trainer.sample(float(split.times[k]), len(split.frames[k]), HELD_OUT_SEED_OFFSET + seed),
                       dtype=np.float64)
        out["samples"][k] = X
        out["held_out"][k] = evaluate(X, split.frames[k], split.weights[k])
    if retained:
        for i in split.retained:
            X = np.asarray(trainer.sample(float(split.times[i]), len(split.frames[i]), RETAINED_SEED_OFFSET + seed),
                           dtype=np.float64)
            out["samples"][i] = X
            out["retained_w1"][i] = evaluate(X, split.frames[i], split.weights[i], metrics=("W1",))["W1"]
    return out


def _points(x: np.ndarray) -> np.ndarray:
    pts = np.asarray(x, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[:, None]
    if pts.ndim != 2 or len(pts) == 0:
        raise ValueError(f"points must be a non-empty (n, d) array; got shape {pts.shape}")
    return pts


def _probabilities(w: np.ndarray) -> np.ndarray:
    """Non-negative weights clipped at zero and divided by their sum."""
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    s = float(w.sum())
    if np.any(w < -1e-9) or not (math.isfinite(s) and s > 0.0):
        raise ValueError("weights must be non-negative with a positive, finite sum")
    return np.clip(w, 0.0, None) / s


def _emd(a: np.ndarray, b: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Exact optimal coupling argmin_P <P, C> of (a, b). When POT stops at its iteration cap, the solve is repeated
    once with 20 times the cap; RuntimeError if that also stops early (the coupling would not be optimal)."""
    plan, capped = _emd_once(a, b, C, EMD_ITERMAX)
    if not capped:
        return plan
    logger.warning("exact EMD reached numItermax=%d on a %s cost matrix; retrying with %d", EMD_ITERMAX, C.shape,
                   20 * EMD_ITERMAX)
    plan, capped = _emd_once(a, b, C, 20 * EMD_ITERMAX)
    if capped:
        raise RuntimeError(f"exact EMD did not converge at numItermax={20 * EMD_ITERMAX} on a {C.shape} cost matrix")
    return plan


def _emd_once(a: np.ndarray, b: np.ndarray, C: np.ndarray, itermax: int) -> tuple[np.ndarray, bool]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = ot.lp.emd(np.ascontiguousarray(a), np.ascontiguousarray(b), np.ascontiguousarray(C),
                         numItermax=int(itermax), check_marginals=False)
    return plan, any("numItermax" in str(w.message) for w in caught)


def _mmd(X: np.ndarray, Y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Squared MMD averaged over the bandwidths MMD_MULTIPLIERS x std(Y) (1 if Y has no spread)."""
    sig = float(np.std(Y)) or 1.0
    Dxx = cdist(X, X, "sqeuclidean")
    Dyy = cdist(Y, Y, "sqeuclidean")
    Dxy = cdist(X, Y, "sqeuclidean")
    tot = 0.0
    for s in tuple(sig * m for m in MMD_MULTIPLIERS):
        g = 1.0 / (2.0 * s * s)
        tot += (a @ np.exp(-g * Dxx) @ a + b @ np.exp(-g * Dyy) @ b
                - 2.0 * a @ np.exp(-g * Dxy) @ b)
    return float(tot / len(MMD_MULTIPLIERS))


# ---------------------------------------------------------------------------------------------------- plots
def plot_figure(paths: np.ndarray | None, path_times: np.ndarray | None,
                frames: Sequence[np.ndarray] | None = None, held_out: Sequence[int] = (), *, title: str = "",
                figsize: tuple[float, float] = (10, 8), alpha: float = 0.6, max_paths: int | None = 200,
                ax: Axes | None = None) -> Figure:
    """Generator paths coloured by time (viridis) with the observed snapshots on top (gist_heat, as in WLR).

    Args:
        paths: (Q, P, d) particle paths at the times `path_times` (e.g. trainer.sample_paths), or None to draw
            only the snapshots.
        path_times: (Q,) times of the paths, Q >= 2 (ignored without paths).
        frames: observed snapshots (n_t, d); snapshot i is labelled "Timestep i+1".
        held_out: withheld snapshots, drawn as hollow circles.
        title: bold title. figsize: size of a new figure. alpha: opacity of the paths.
        max_paths: the first max_paths paths are drawn (None: all).
        ax: axes to draw into (default: a new figure).

    The first two coordinates are shown, at most 2000 points per snapshot (a fixed random subset). Returns the
    figure; plt.show() is not called.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    if paths is not None:
        P = np.asarray(paths, dtype=np.float64)
        t = np.asarray(path_times, dtype=np.float64).reshape(-1)
        if P.ndim != 3 or P.shape[0] != len(t) or len(t) < 2 or P.shape[2] < 2:
            raise ValueError(f"paths must be (Q, P, d) with Q = len(path_times) >= 2 and d >= 2; got {P.shape} "
                             f"and {len(t)} times")
        Y = P[:, :max_paths, :2]
        segments = np.stack([Y[:-1], Y[1:]], axis=2).transpose(1, 0, 2, 3).reshape(-1, 2, 2)
        colours = np.tile(0.5 * (t[:-1] + t[1:]), Y.shape[1])
        keep = np.isfinite(segments).all(axis=(1, 2))
        norm = mpl.colors.Normalize(vmin=t[0], vmax=t[-1] if t[-1] > t[0] else t[0] + 1.0)
        cmap = mpl.colormaps["viridis"]
        lines = LineCollection(segments[keep], cmap=cmap, norm=norm, linewidths=0.8, alpha=alpha, zorder=1,
                               rasterized=True)
        lines.set_array(colours[keep])
        ax.add_collection(lines)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Time")
    if frames is not None:
        heat = mpl.colormaps["gist_heat"]
        held = {int(k) for k in held_out}
        for i, X in enumerate(frames):
            X = _subsample(np.asarray(X, dtype=np.float64), PLOT_MAX_POINTS, seed=i)
            if X.ndim != 2 or X.shape[1] < 2:
                raise ValueError(f"snapshot {i} must be (n, d) with d >= 2; got {X.shape}")
            colour = heat(i / len(frames))
            if i in held:
                ax.scatter(X[:, 0], X[:, 1], s=12, facecolors="none", edgecolors=[colour], linewidths=0.8,
                           label=f"Timestep {i + 1} (held out)", zorder=3)
            else:
                ax.scatter(X[:, 0], X[:, 1], s=12, color=[colour], alpha=0.8, linewidths=0,
                           label=f"Timestep {i + 1}", zorder=2)
        ax.legend(markerscale=2.0)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.autoscale_view()
    return fig


def plot_snapshots(frames: Sequence[np.ndarray], generated: Sequence[np.ndarray | None], times: Sequence[float],
                   held_out: Sequence[int] = (), *, title: str = "") -> Figure:
    """Observed (top row, grey) and generated (bottom row, blue) snapshots, one column per observed time.

    Args:
        frames: the T observed snapshots (n_t, d).
        generated: T arrays of generated samples, None for an empty cell (e.g. evaluate_split's "samples").
        times: the T snapshot times (column titles).
        held_out: withheld snapshots (titles and frames in red).
        title: figure title.

    The panels share square limits; first two coordinates, at most 2000 points per panel.
    """
    T = len(frames)
    if len(generated) != T or len(times) != T:
        raise ValueError(f"{T} snapshots, {len(generated)} generated entries and {len(times)} times")
    obs = [_subsample(np.asarray(x, dtype=np.float64)[:, :2], PLOT_MAX_POINTS, seed=i) for i, x in enumerate(frames)]
    gen = [None if g is None else _subsample(np.asarray(g, dtype=np.float64)[:, :2], PLOT_MAX_POINTS, seed=i)
           for i, g in enumerate(generated)]
    xlim, ylim = _square_limits(obs + [g for g in gen if g is not None])
    held = {int(k) for k in held_out}
    fig, axes = plt.subplots(2, T, figsize=(1.6 * T + 0.6, 3.6), squeeze=False, layout="constrained")
    for i in range(T):
        top, bottom = axes[0, i], axes[1, i]
        top.scatter(obs[i][:, 0], obs[i][:, 1], s=3, color="0.35", alpha=0.6, linewidths=0, rasterized=True)
        if gen[i] is not None:
            bottom.scatter(gen[i][:, 0], gen[i][:, 1], s=3, color=BLUE, alpha=0.6, linewidths=0, rasterized=True)
        top.set_title(f"t = {float(times[i]):g}" + (" (held out)" if i in held else ""), fontsize=8,
                      color=RED if i in held else "black")
        for ax in (top, bottom):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(RED if i in held else "0.5")
    axes[0, 0].set_ylabel("observed", fontsize=8)
    axes[1, 0].set_ylabel("generated", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    return fig


def _subsample(X: np.ndarray, n_max: int, seed: int = 0) -> np.ndarray:
    """At most n_max rows of X: a fixed random subset in the original order (X itself if it is small)."""
    if len(X) <= n_max:
        return X
    return X[np.sort(np.random.default_rng(seed).choice(len(X), size=n_max, replace=False))]


def _square_limits(arrays: Sequence[np.ndarray], pad: float = 0.08
                   ) -> tuple[tuple[float, float], tuple[float, float]]:
    """A square box around the finite 2-D points of `arrays`, widened by `pad` of its side."""
    pts = np.concatenate([a.reshape(-1, 2) for a in arrays if len(a)], axis=0)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * max(float((hi - lo).max()), 1e-9 * max(1.0, float(np.abs(hi).max()))) * (1.0 + 2.0 * pad)
    return (float(centre[0] - half), float(centre[0] + half)), (float(centre[1] - half), float(centre[1] + half))


# ---------------------------------------------------------------------------------------------------- files
def save_json(path: str | Path, obj: Any) -> None:
    """Write `obj` as indented JSON: numpy arrays and scalars become Python values, tuples lists, keys strings,
    NaN and inf null (other objects are written as their str)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, allow_nan=False, default=str) + "\n")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj
