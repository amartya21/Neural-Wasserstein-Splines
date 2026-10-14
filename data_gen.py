"""
Loads or generates the datasets of NWSP as time-ordered snapshots (float64 arrays, one row per point):
 - get_dataset : one of the 13 datasets by name (three toy datasets and the ten benchmark datasets of the paper)
 - split_dataset : withholds interior snapshots; the embryoid-body data are standardised on the retained ones
 - standardize : per-coordinate standardisation fitted on the retained snapshots, applied to every snapshot
 - diverging_gaussians, petal, converging_gaussian, circular_gaussian : generators of the four synthetic benchmarks
 - group_sample : the per-time subsampling rule of the WLR CSV loaders

The four generators are numpy ports of data_gen.py of the Wasserstein Lane-Riesenfeld (WLR) repository (MIT licence,
Copyright (c) 2025 Amartya Banerjee). They reproduce its frames bit for bit: the same draws in the same order on a
private np.random.RandomState(seed), so the global random state is never touched. The other datasets are read from
the files in Data/ (sources, licences and checksums in Data/README.md).
"""
from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

DATA_DIR: Path = Path(__file__).resolve().parent / "Data"
TOY_DATASETS: tuple[str, ...] = ("circular", "converging", "expanding")
BENCHMARK_DATASETS: tuple[str, ...] = ("diverging_gaussians", "diamonds", "converging_gaussian", "circular_gaussian",
                                       "dyngen_tree", "dyngen_cycle", "eb_phate", "bsbm_mouse", "bsbm_veres",
                                       "citeseq")
DATASETS: tuple[str, ...] = TOY_DATASETS + BENCHMARK_DATASETS
ALIASES: dict[str, str] = {"petal": "diamonds"}


class _Info(NamedTuple):
    label: str
    T: int
    d: int
    weights: str      # truth-weight rule of the scores, see _truth_weights
    file: str | None  # bundled file in Data/ (None: generated)


_REGISTRY: dict[str, _Info] = {
    "circular": _Info("Circular (toy)", 5, 2, "uniform", "toy_data.npz"),
    "converging": _Info("Converging (toy)", 5, 2, "uniform", "toy_data.npz"),
    "expanding": _Info("Expanding (toy)", 5, 2, "uniform", "toy_data.npz"),
    "diverging_gaussians": _Info("Diverging Gaussian", 4, 2, "official", None),
    "diamonds": _Info("Petal", 5, 2, "official", None),
    "converging_gaussian": _Info("Converging Gaussian", 4, 2, "official", None),
    "circular_gaussian": _Info("Circular Gaussian", 10, 2, "official", None),
    "dyngen_tree": _Info("Dyngen Tree", 7, 10, "official", "dyngen_tree.csv"),
    "dyngen_cycle": _Info("Dyngen Cycle", 15, 10, "official", "dyngen_cycle.csv"),
    "eb_phate": _Info("Embryoid body (PHATE)", 5, 2, "uniform", "eb_phate.npz"),
    "bsbm_mouse": _Info("Mouse hematopoiesis", 3, 2, "uniform", "mouse_hematopoiesis.csv"),
    "bsbm_veres": _Info("Veres beta-cell", 8, 30, "uniform", "Veres_alltime.csv.gz"),
    "citeseq": _Info("CITE-seq", 4, 100, "official", "citeseq.csv.gz"),
}
_CSV_ROWS_PER_TIME: dict[str, int] = {"dyngen_tree": 319, "dyngen_cycle": 52, "citeseq": 1000}
_STANDARDIZED: tuple[str, ...] = ("eb_phate",)


@dataclass(frozen=True, eq=False)
class Snapshots:
    """Snapshots of a population.

    Attributes:
        name, label: dataset key and display name.
        times: (T,) float64, strictly increasing: 0, 0.25, ..., 1 (toy data) or 0, 1, ..., T - 1 (benchmarks).
        frames: T float64 C-contiguous arrays (n_t, d), one row per point.
        weights: T arrays (n_t,), the weights of the observed points in the scores (uniform).
        standardize: True if every split is standardised with the statistics of its retained snapshots
            (the embryoid-body data).
    """

    name: str
    label: str
    times: np.ndarray
    frames: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    standardize: bool = False

    @property
    def T(self) -> int:
        """Number of snapshots."""
        return len(self.frames)

    @property
    def d(self) -> int:
        """Dimension."""
        return int(self.frames[0].shape[1])

    @property
    def sizes(self) -> list[int]:
        """Points per snapshot."""
        return [int(f.shape[0]) for f in self.frames]

    def __repr__(self) -> str:
        return f"Snapshots({self.name!r}, T={self.T}, d={self.d}, sizes={self.sizes})"


@dataclass(frozen=True, eq=False)
class Split:
    """Snapshots with some interior snapshots withheld from the fit.

    Attributes:
        data: the dataset.
        held_out: sorted withheld interior indices (empty: every snapshot is fitted).
        retained: the other indices, sorted.
        frames: all T snapshots in training coordinates: standardised with the statistics of the retained snapshots
            if data.standardize (the withheld ones too), otherwise data.frames.
        whitening: (mean, std), both (d,), of the standardisation, or None; data coordinates are x * std + mean.
    """

    data: Snapshots
    held_out: tuple[int, ...]
    retained: tuple[int, ...]
    frames: tuple[np.ndarray, ...]
    whitening: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def times(self) -> np.ndarray:
        """All snapshot times."""
        return self.data.times

    @property
    def weights(self) -> tuple[np.ndarray, ...]:
        """Truth weights of all snapshots (the standardisation does not change them)."""
        return self.data.weights

    @property
    def train_times(self) -> np.ndarray:
        """Times of the retained snapshots."""
        return self.data.times[list(self.retained)]

    @property
    def train_frames(self) -> list[np.ndarray]:
        """Retained snapshots in training coordinates."""
        return [self.frames[i] for i in self.retained]

    def __repr__(self) -> str:
        return (f"Split({self.data.name!r}, held_out={list(self.held_out)}, retained={list(self.retained)}, "
                f"standardized={self.whitening is not None})")


def get_dataset(name: str, seed: int = 0, data_dir: str | Path | None = None) -> Snapshots:
    """Load or generate a dataset.

    Args:
        name: a key of DATASETS or the alias 'petal' (case-insensitive).
        seed: seed of the four synthetic generators and of the subsampling of the three WLR CSV files
            (dyngen_tree, dyngen_cycle, citeseq); 0 gives the paper's data. The toy, BranchSBM and embryoid-body
            data are fixed files (the seed is not used).
        data_dir: directory of the bundled files (default: Data/ next to this file).

    Returns:
        The snapshots, with times 0, 0.25, ..., 1 (toy data) or 0, 1, ..., T - 1 (benchmarks).

    Raises:
        ValueError: unknown name. FileNotFoundError: a bundled file is missing (the message names it).
    """
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in _REGISTRY:
        known = ", ".join(list(DATASETS) + [f"{a} (= {b})" for a, b in ALIASES.items()])
        raise ValueError(f"unknown dataset {name!r}; known datasets: {known}")
    info = _REGISTRY[key]
    root = DATA_DIR if data_dir is None else Path(data_dir).expanduser()
    if key in TOY_DATASETS:
        times, frames = _read_toy(_require(root / info.file, key), key)
    else:
        frames = _benchmark_frames(key, int(seed), root)
        times = np.arange(len(frames), dtype=np.float64)
    if len(frames) != info.T or frames[0].shape[1] != info.d:
        raise ValueError(f"{key}: loaded {len(frames)} snapshots of dimension {frames[0].shape[1]}, "
                         f"expected T = {info.T}, d = {info.d}")
    weights = tuple(_truth_weights(len(f), info.weights) for f in frames)
    return Snapshots(key, info.label, times, tuple(frames), weights, key in _STANDARDIZED)


def split_dataset(data: Snapshots, held_out: Sequence[int] = ()) -> Split:
    """Withhold the interior snapshots `held_out` (distinct indices 0 < k < T - 1; empty: fit every snapshot).

    If data.standardize, every snapshot is standardised with the mean and standard deviation of the retained
    snapshots only (`standardize`), so no statistic of a withheld snapshot enters the training coordinates.
    """
    try:
        held = tuple(sorted(operator.index(k) for k in held_out))
    except TypeError:
        raise ValueError(f"held_out must be a sequence of integer indices; got {held_out!r}") from None
    if len(set(held)) != len(held):
        raise ValueError(f"held-out indices must be distinct; got {list(held)}")
    bad = [k for k in held if not 0 < k < data.T - 1]
    if bad:
        raise ValueError(f"held-out indices must be interior (1..{data.T - 2}) for {data.name!r}; got {bad}")
    retained = tuple(i for i in range(data.T) if i not in held)
    if not data.standardize:
        return Split(data, held, retained, data.frames, None)
    frames, mean, std = standardize(data.frames, retained)
    return Split(data, held, retained, tuple(frames), (mean, std))


def standardize(frames: Sequence[np.ndarray], retained: Sequence[int]
                ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Per-coordinate standardisation fitted on the pooled retained snapshots and applied to every snapshot.

    mean and std (population, ddof 0) of the concatenated snapshots `retained`; coordinates with std <= 1e-12 keep
    scale 1. Returns ([(x - mean) / std for x in frames], mean, std); data coordinates are x * std + mean.
    """
    pool = np.concatenate([np.asarray(frames[i], dtype=np.float64) for i in retained], 0)
    mu = pool.mean(0)
    sd = pool.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return [(np.asarray(c, dtype=np.float64) - mu) / sd for c in frames], mu, sd


# ---------------------------------------------------------------------------------------------------- loaders
def _truth_weights(n: int, rule: str) -> np.ndarray:
    """Weights of an observed snapshot of n points: "uniform" = full(n, 1/n); "official" = the same vector divided
    by its sum, as the WLR loaders normalise it (it differs from "uniform" in the last bit for some n, e.g. 19, 100,
    200 and 1000)."""
    w = np.full(n, 1.0 / n)
    return w / w.sum() if rule == "official" else w


def _require(path: Path, name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"data file {path} of dataset {name!r} not found; the bundled files are in Data/ of "
                                f"the repository (pass data_dir= if they are elsewhere)")
    return path


def _benchmark_frames(name: str, seed: int, root: Path) -> list[np.ndarray]:
    if name == "diverging_gaussians":
        return diverging_gaussians(200, seed=seed)
    if name == "diamonds":
        return petal(seed=seed)
    if name == "converging_gaussian":
        return converging_gaussian(seed=seed)
    if name == "circular_gaussian":
        return circular_gaussian(seed=seed)
    path = _require(root / _REGISTRY[name].file, name)
    if name in _CSV_ROWS_PER_TIME:
        return _read_wlr_csv(path, _CSV_ROWS_PER_TIME[name], seed)
    if name == "eb_phate":
        return _read_eb_npz(path)
    return _read_bsbm_csv(path)


def _read_toy(path: Path, name: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """`times` (5,) and the (5, 256, 2) array `name` of toy_data.npz (see Data/README.md for how it was made)."""
    with np.load(path, allow_pickle=False) as z:
        times = np.asarray(z["times"], dtype=np.float64)
        clouds = np.asarray(z[name], dtype=np.float64)
    return times, [np.ascontiguousarray(c) for c in clouds]


def _read_csv(path: Path) -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(path)


def _read_wlr_csv(path: Path, n_per_time: int, seed: int = 0) -> list[np.ndarray]:
    """CSV of the WLR repository (dyngen_tree, dyngen_cycle, citeseq): first column `samples` = time label, the other
    columns = coordinates. Each time label (ascending) is subsampled to at most `n_per_time` rows with `group_sample`
    on RandomState(seed), which reproduces the repository's subsample_dataframe(df, n_per_time, seed)."""
    df = _read_csv(path)
    cols = list(df.columns)
    if not cols or cols[0] != "samples":
        raise ValueError(f"{path}: the first column must be 'samples' (the time label); got {cols[:3]}")
    labels = df["samples"].to_numpy()
    X = df[cols[1:]].to_numpy(dtype=np.float64)
    rows = group_sample(labels, n_per_time, np.random.RandomState(seed))
    return [np.ascontiguousarray(X[r]) for r in rows]


def _read_bsbm_csv(path: Path) -> list[np.ndarray]:
    """CSV released with BranchSBM (mouse_hematopoiesis, Veres): column `samples` = time label, the other columns =
    coordinates; every row is used, snapshots in ascending label order, rows in file order."""
    df = _read_csv(path)
    if "samples" not in df.columns:
        raise ValueError(f"{path}: no 'samples' column (the time label)")
    xcols = [c for c in df.columns if c != "samples"]
    labels = df["samples"].astype(float).to_numpy()
    X = df[xcols].to_numpy(dtype=np.float64)
    return [np.ascontiguousarray(X[labels == v]) for v in np.unique(labels)]


def _read_eb_npz(path: Path) -> list[np.ndarray]:
    """Embryoid-body PHATE coordinates (`phate`) grouped by `sample_labels` (ascending), rows in file order, no
    normalisation (every split is standardised by split_dataset)."""
    with np.load(path, allow_pickle=False) as z:
        coords = np.asarray(z["phate"], dtype=np.float64)
        labels = np.asarray(z["sample_labels"]).reshape(-1)
    if len(labels) != len(coords):
        raise ValueError(f"{path}: {len(labels)} labels for {len(coords)} points")
    return [np.ascontiguousarray(coords[np.nonzero(labels == v)[0]]) for v in np.unique(labels)]


def group_sample(labels: np.ndarray, k: int, rs: np.random.RandomState) -> list[np.ndarray]:
    """Row indices of a per-label subsample of at most k rows, labels in sorted order.

    For each label (ascending) with rows idx (in file order) this returns idx[rs.permutation(len(idx))[:min(len(idx),
    k)]], which is what pandas df.groupby(label).apply(lambda x: x.sample(min(len(x), k))) returns after
    np.random.seed(seed) when rs = RandomState(seed).
    """
    labels = np.asarray(labels).reshape(-1)
    out = []
    for v in np.unique(labels):
        idx = np.nonzero(labels == v)[0]
        out.append(idx[rs.permutation(len(idx))[:min(len(idx), int(k))]])
    return out


# ---------------------------------------------------------------------------------------------------- generators
def _frame(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def diverging_gaussians(n_samples: int = 200, seed: int = 0) -> list[np.ndarray]:
    """Four 2-D Gaussian mixtures that split over time (WLR's get_diverging_gaussians).

    t = 0: N(0, C1); t = 1: two equal components at (2, -1) and (2, 2); t = 2: 80 % / 10 % / 10 % components at
    (5, 0), (5, 1.5), (5, 2.5); t = 3: 20 % / 80 % at (8, 1), (8, 3). Each covariance is a matrix S squared as S S^T
    (half of it for the minor components of t = 2). numpy's multivariate_normal factorises the covariances through
    LAPACK / BLAS, so the last bits of these frames depend on the BLAS kernels (Data/README.md).
    """
    rs = np.random.RandomState(seed)
    n = n_samples

    cov1 = 0.1 * np.eye(2)
    cov1 = cov1 @ cov1.T
    sample1 = rs.multivariate_normal(np.array([0, 0]), cov1, n).T

    cov21 = 0.1 * np.array([[1, 2], [2, 1]])
    cov21 = cov21 @ cov21.T
    sample21 = rs.multivariate_normal(np.array([2, -1]), cov21, n // 2).T
    sample22 = rs.multivariate_normal(np.array([2, 2]), cov21, n // 2).T
    sample2 = np.concatenate((sample21, sample22), axis=1)

    cov31 = 0.1 * np.array([[1, -3], [-3, 1]])
    cov31 = cov31 @ cov31.T
    sample31 = rs.multivariate_normal(np.array([5, 0]), cov31, int(n * 0.8)).T
    sample32 = rs.multivariate_normal(np.array([5, 1.5]), 0.5 * cov31, int(n * 0.1)).T
    sample33 = rs.multivariate_normal(np.array([5, 2.5]), 0.5 * cov31, int(n * 0.1)).T
    sample3 = np.concatenate((sample31, sample32, sample33), axis=1)

    cov41 = 0.1 * np.array([[3, 0.1], [0.1, 2]])
    cov41 = cov41 @ cov41.T
    sample41 = rs.multivariate_normal(np.array([8, 1]), cov41, int(n * 0.2)).T
    sample42 = rs.multivariate_normal(np.array([8, 3]), cov41, int(n * 0.8)).T
    sample4 = np.concatenate((sample41, sample42), axis=1)

    return [_frame(s.T) for s in (sample1, sample2, sample3, sample4)]


# Petal follows the petal dataset of MIOFlow (https://github.com/KrishnaswamyLab/MIOFlow) as included in data_gen.py
# of the WLR repository (https://github.com/amartya21/Wasserstein-Trajectory-Inference).
def _construct_petals(points_per_petal: int, petal_width: float, direction: str) -> np.ndarray:
    """Four petals along one axis: piecewise-linear width profile, mirrored in both coordinates."""
    n_side = points_per_petal // 2
    ax1 = np.concatenate([np.linspace(0, petal_width, n_side // 2), np.linspace(petal_width, 0, n_side // 2)])
    ax2 = np.linspace(0, 1, n_side)
    pts = np.vstack((ax1, ax2)).T if direction == "y" else np.vstack((ax2, ax1)).T
    pts = np.vstack((pts, -pts))
    return np.vstack((pts, np.vstack((pts[:, 0], -pts[:, 1])).T))


def petal(points_per_petal: int = 500, petal_width: float = 0.25, colors: int = 5, scale_factor: float = 30.0,
          seed: int = 0) -> list[np.ndarray]:
    """Petal snapshots, as WLR's get_diamonds(points_per_petal, petal_width, colors, scale_factor, seed=seed).

    Nested petal shapes whose L-infinity shells are the snapshots. The two petal crosses (along y and along x) get
    Gaussian noise randn / scale_factor. A point gets label v of the innermost L-infinity box max(|x_1|, |x_2|) <= th_v
    that contains it (v from `colors` down to 1 with thresholds linspace(1, 1 / (colors + 1), colors); points outside
    every box keep `colors`); every label is subsampled to the smallest label count with group_sample on the same
    random state. Snapshots are in label order: label 1 (the centre) is time 0.

    Returns:
        `colors` snapshots of equal size (321 points each for the defaults and seed 0).
    """
    rs = np.random.RandomState(seed)
    upper = _construct_petals(points_per_petal, petal_width, "y")
    lower = _construct_petals(points_per_petal, petal_width, "x")
    data = np.vstack((upper, lower))
    data = data + rs.randn(*data.shape) / scale_factor
    labels = np.full(len(data), float(colors))
    for v, th in zip(np.linspace(colors, 1, colors), np.linspace(1, 1 / (colors + 1), colors)):
        mask = (np.abs(data[:, 0]) <= th) & (np.abs(data[:, 1]) <= th)
        labels[mask] = v
    _, counts = np.unique(labels, return_counts=True)
    rows = group_sample(labels, int(counts.min()), rs)
    return [np.ascontiguousarray(data[r]) for r in rows]


def _circle_supp(R: float, n: int) -> np.ndarray:
    """The origin plus rings of radii linspace(0, R, n), ring i with floor(4 n Rs_i / R) points."""
    B = 4 * n / R
    Rs = np.linspace(0, R, n)
    x = [0]
    y = [0]
    for i in range(n):
        num_points = int(np.floor(B * Rs[i]))
        if num_points == 0:
            continue
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x.extend(Rs[i] * np.cos(theta))
        y.extend(Rs[i] * np.sin(theta))
    return np.vstack((x, y)).T


def _bump(center: Sequence[float] | np.ndarray, n: int, B: float, sampling_factor: int) -> np.ndarray:
    return _circle_supp(1.5 * B, n // sampling_factor) + center


def converging_gaussian(seed: int = 0) -> list[np.ndarray]:
    """Deterministic ring grids that split and merge again (WLR's get_dataset("converging_gaussian"), adapted there
    from Clancy and Suarez, arXiv:2203.15728).

    The only random draw is the thinning factor RandomState(seed).randint(1, 6) (5 for seed 0, giving snapshots of
    19, 57, 38 and 19 points): t = 0 one bump at the origin, t = 1 three bumps around (1, 0), t = 2 two bumps at
    (1.5, +-1), t = 3 one bump at (2, 0).
    """
    sampling_factor = int(np.random.RandomState(seed).randint(1, 6))
    n, B = 16, 0.1
    x1 = _bump([0, 0], n, 2 * B, sampling_factor)
    x2 = np.vstack([_bump(np.sqrt(2) / 2 * np.array([1, 1]), n, B, sampling_factor),
                    _bump(np.sqrt(2) / 2 * np.array([1, -1]), n, B, sampling_factor),
                    _bump([1, 0], n, B, sampling_factor)])
    x3 = np.vstack([_bump([1.5, 1], n, B, sampling_factor), _bump([1.5, -1], n, B, sampling_factor)])
    x4 = _bump([2, 0], n, 2 * B, sampling_factor)
    return [_frame(x) for x in (x1, x2, x3, x4)]


def circular_gaussian(number_mid: int = 10, radius: float = 10.0, n_samples: int = 100, noise_scale: float = 0.5,
                      seed: int = 0) -> list[np.ndarray]:
    """Gaussian clouds N(m_i, noise_scale I_2) around a circle, m_i = radius (cos s_i, sin s_i) with
    s = linspace(0, 2 pi - 0.5, number_mid) (WLR's get_circular_gaussian)."""
    rs = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi - 0.5, number_mid)
    M = np.vstack((radius * np.cos(t), radius * np.sin(t)))
    cov = noise_scale * np.eye(2)
    return [_frame(rs.multivariate_normal(M[:, i], cov, n_samples)) for i in range(number_mid)]
