#!/usr/bin/env python
"""Fit NWSP or NWSP-N to one dataset, score the withheld snapshots and save the outputs.

Examples:
    python run_nwsp.py --dataset circular --held_out 2 --quickstart
    python run_nwsp.py --dataset diamonds --held_out 2 --method NWSP-N
    python run_nwsp.py --dataset dyngen_cycle --lambda_acc 1e-6

Outputs in results/<method>_<dataset>[_h<k>...][_lam<L>][_qs][_steps<N>][_<device>]_s<seed>/ (or --output):
metrics.json (scores and settings), samples.npz (scored samples and generator paths), trajectories.png and
checkpoint.pt (nwsp.load_model). Exit codes: 0 success; 1 the fit diverged or produced non-finite samples
(metrics.json records the reason), or an unexpected error (traceback); 2 invalid arguments or data.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

DATASET_NAMES = ("circular, converging, expanding (toy data); diverging_gaussians, diamonds (alias petal), "
                 "converging_gaussian, circular_gaussian, dyngen_tree, dyngen_cycle, eb_phate, bsbm_mouse, "
                 "bsbm_veres, citeseq")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
OUTPUT_FILES = ("metrics.json", "samples.npz", "trajectories.png", "checkpoint.pt")
PATH_SEED_OFFSET = 3000            # plotted generator paths: seed 3000 + training seed
N_PATHS, N_PATH_TIMES = 256, 101
# --quickstart settings of single toy datasets: the two merging blobs of `converging` need 1000 steps
QUICKSTART_DATASETS = {"converging": {"steps": 1000, "warmup_steps": 200, "log_every": 25}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], allow_abbrev=False,
                                     epilog="README.md lists the paper's settings and penalty weights.")
    parser.add_argument("--dataset", required=True, metavar="NAME", help=f"dataset: {DATASET_NAMES}")
    parser.add_argument("--method", choices=("NWSP", "NWSP-N"), default="NWSP",
                        help="NWSP, or NWSP-N with learned potentials (default: %(default)s)")
    parser.add_argument("--held_out", type=int, nargs="+", metavar="K",
                        help="interior snapshots withheld from the fit and scored (default: none, every snapshot "
                             "is fitted)")
    parser.add_argument("--lambda_acc", type=float, metavar="L",
                        help="weight of the acceleration penalty; 0 gives the plain spline fit (default: 1e-4, "
                             "1e-3 with --quickstart)")
    parser.add_argument("--steps", type=int, metavar="N",
                        help="training steps (default: 3000; with --quickstart 200, 1000 for converging); the "
                             "warm-up keeps its share")
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="training seed (default: %(default)s)")
    parser.add_argument("--device", default="auto",
                        help="auto (CUDA when available and d > 10), cpu, cuda or cuda:N (default: %(default)s)")
    parser.add_argument("--threads", type=int, default=1, metavar="T",
                        help="torch / BLAS threads (default: %(default)s, as for the README's reference scores)")
    parser.add_argument("--quickstart", action="store_true",
                        help="small, fast settings meant for the toy data (NWSPConfig.quickstart(); 1000 steps for "
                             "converging)")
    parser.add_argument("--held_out_only", action="store_true",
                        help="skip the W1 diagnostic at the retained snapshots (minutes on bsbm_veres)")
    parser.add_argument("--output", type=Path, metavar="DIR",
                        help="output directory (default: results/<method>_<dataset>[_h<k>...][_lam<L>][_qs]"
                             "[_steps<N>][_<device>]_s<seed>, the optional parts when --held_out, --lambda_acc, "
                             "--quickstart, --steps or --device (other than auto) are given); files of an earlier "
                             "run in it are replaced")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be at least 1")
    for var in THREAD_VARIABLES:  # before numpy and torch are imported
        os.environ[var] = str(args.threads)
    t_start = time.time()

    import numpy as np

    from data_gen import get_dataset, split_dataset

    try:
        data = get_dataset(args.dataset)
        split = split_dataset(data, args.held_out or ())
    except (ValueError, FileNotFoundError) as exc:
        return _error(exc)
    if args.device == "cpu" or (args.device == "auto" and data.d <= 10):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # a CPU run never opens a CUDA context

    import matplotlib
    import torch

    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nwsp import NWSPConfig, NWSPNConfig, NWSPNTrainer, NWSPTrainer
    from utils import METRICS, evaluate_split, plot_figure, save_json

    logger = logging.getLogger("run_nwsp")
    neural = args.method == "NWSP-N"
    config_cls, trainer_cls = (NWSPNConfig, NWSPNTrainer) if neural else (NWSPConfig, NWSPTrainer)
    held = list(split.held_out)
    out = args.output if args.output is not None else Path("results") / _run_name(args, data.name, held)
    try:
        config = (config_cls.quickstart(**QUICKSTART_DATASETS.get(data.name, {})) if args.quickstart
                  else config_cls())
        if args.steps is not None:
            config = replace(config, warmup_steps=round(config.warmup_steps * args.steps / config.steps),
                             steps=args.steps)
        changes = {"seed": args.seed, "device": args.device}
        if args.lambda_acc is not None:
            changes["lambda_acc"] = args.lambda_acc
        config = replace(config, **changes)
        trainer = trainer_cls(split.train_times, split.train_frames, config)
        out.mkdir(parents=True, exist_ok=True)
        for f in OUTPUT_FILES:
            (out / f).unlink(missing_ok=True)
    except (ValueError, OSError) as exc:
        return _error(exc)

    record = {"method": args.method, "dataset": data.name, "label": data.label, "held_out": held, "seed": args.seed,
              "device": str(trainer.device), "train_seconds": None, "diverged": None, "scores": {},
              "held_out_mean": None, "retained_w1": {}, "retained_fit_W1": None}
    t_fit = time.perf_counter()
    try:
        trainer.fit()
    except (FloatingPointError, torch.linalg.LinAlgError) as exc:
        kind = "LinAlgError" if isinstance(exc, torch.linalg.LinAlgError) else type(exc).__name__
        record.update(train_seconds=time.perf_counter() - t_fit, diverged=f"{kind}: {exc}",
                      scores={str(k): dict.fromkeys(METRICS, math.nan) for k in held},
                      config=trainer.config.to_dict())
        logger.warning("the fit diverged (%s); scores are NaN", record["diverged"])
        save_json(out / "metrics.json", record)
        try:
            trainer.save(out / "checkpoint.pt", whitening=split.whitening)
        except Exception:  # the metrics of a diverged fit are written in any case
            logger.exception("could not save the checkpoint of the diverged fit")
        print(f"{args.method} on {data.label}: the fit diverged ({record['diverged']})")
        print(f"Outputs: {out}  ({time.time() - t_start:.1f} s)")
        return 1
    train_seconds = trainer.train_seconds if trainer.train_seconds is not None else time.perf_counter() - t_fit

    res = evaluate_split(trainer, split, retained=not args.held_out_only)
    path_times = np.linspace(split.times[0], split.times[-1], N_PATH_TIMES)
    paths = trainer.sample_paths(path_times, N_PATHS, PATH_SEED_OFFSET + args.seed)

    scores, retained_w1 = res["held_out"], res["retained_w1"]
    values = [v for sc in scores.values() for v in sc.values()] + list(retained_w1.values())
    finite = all(math.isfinite(v) for v in values)
    held_mean = {m: float(np.mean([scores[k][m] for k in held])) for m in METRICS} if held else None
    retained_mean = float(np.mean(list(retained_w1.values()))) if retained_w1 else None
    record.update(train_seconds=train_seconds, diverged=None if finite else "non-finite samples",
                  scores={str(k): sc for k, sc in scores.items()}, held_out_mean=held_mean,
                  retained_w1={str(i): w for i, w in retained_w1.items()}, retained_fit_W1=retained_mean,
                  config=trainer.config.to_dict())
    trainer.save(out / "checkpoint.pt", whitening=split.whitening)
    arrays = {"times": np.asarray(split.times, dtype=np.float64), "held_out": np.asarray(held, dtype=np.int64),
              "path_times": path_times, "paths": np.asarray(paths, dtype=np.float32)}
    arrays.update({f"generated_{i}": np.asarray(X, dtype=np.float64) for i, X in sorted(res["samples"].items())})
    if split.whitening is not None:
        arrays["whiten_mean"], arrays["whiten_std"] = split.whitening
    np.savez_compressed(out / "samples.npz", **arrays)
    save_json(out / "metrics.json", record)
    try:
        fig = plot_figure(paths, path_times, split.frames, held, title=f"{args.method} on {data.label}")
        fig.savefig(out / "trajectories.png", dpi=150)
        plt.close(fig)
    except Exception:  # a failed figure never loses a finished fit
        logger.exception("plotting failed; the other outputs are complete")

    if held:
        held_text = f"held out {held}: " + "  ".join(f"{m} {_fmt(held_mean[m])}" for m in METRICS)
    else:
        held_text = "every snapshot fitted"
    retained_text = "not scored" if args.held_out_only else _fmt(retained_mean)
    print(f"{args.method} on {data.label}: trained in {train_seconds:.1f} s; {held_text}; "
          f"retained-fit W1 {retained_text}")
    print(f"Outputs: {out}  ({time.time() - t_start:.1f} s)")
    return 0 if finite else 1


def _run_name(args: argparse.Namespace, dataset: str, held: list[int]) -> str:
    """Default output directory name: the method, the dataset and every given option that changes the results."""
    name = f"{args.method}_{dataset}"
    if held:
        name += "_h" + "-".join(map(str, held))
    if args.lambda_acc is not None:
        name += f"_lam{args.lambda_acc:g}"
    if args.quickstart:
        name += "_qs"
    if args.steps is not None:
        name += f"_steps{args.steps}"
    if args.device != "auto":
        name += "_" + args.device.split(":")[0]
    return name + f"_s{args.seed}"


def _fmt(x: float | None) -> str:
    return "nan" if x is None else f"{x:.4f}"


def _error(exc: BaseException) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
