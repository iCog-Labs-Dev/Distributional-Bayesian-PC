"""Out-of-distribution evaluation on saved DBPCN checkpoints (rotated MNIST).

Loads one or more `runs/<name>/` checkpoints, evaluates each on a configurable
sweep of MNIST rotations, and writes a per-checkpoint JSON + a unified
markdown / JSON comparison report.

Purely an eval-side script: no JAX-traced changes to `bpcn/`, no retraining.
Reuses `_load_checkpoint` from `experiments.mnist`, the target-free shared-DPC
E-step from `bpcn.evaluation.predict._target_free_frozen`, and the existing
MEAN / MC predictive functions.

Usage
-----
    python -m experiments.ood_eval --help
    python -m experiments.ood_eval `
        --run-dirs runs/cat_mc_10c2,runs/cat_mean_10c2 `
        --rotations 0,45,90 `
        --out-dir reports/ood_rotated_mnist

The angle=0 row is the in-distribution baseline and should reproduce the
saved final-epoch metrics in `history.json` within rounding error -- this is
the primary correctness self-test before reading any OOD numbers.

Rotation
--------
Per-image rotation around the image center via `scipy.ndimage.rotate` with
`reshape=False` and zero padding -- standard rotated-MNIST convention. The
input is reshaped `[N, 784] -> [N, 28, 28]`, rotated per-image, and flattened
back. Angle=0 short-circuits and returns the input unchanged.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from sys import path
from typing import Dict, List, Tuple

path.append(".")

import jax
import jax.numpy as jnp
import numpy as np

from bpcn.configs.base import BaseConfig
from bpcn.data.mnist import load_split
from bpcn.evaluation.predict import (
    _mc_predict_from_frozen,
    _mean_predict_from_frozen,
    _target_free_frozen,
    predictive_entropy,
)
from experiments.mnist import _load_checkpoint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_run_dirs(s: str) -> Tuple[str, ...]:
    parts = [p.strip().rstrip("/").rstrip("\\") for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--run-dirs requires at least one path")
    return tuple(parts)


def _parse_angles(s: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--rotations requires at least one angle")
    try:
        return tuple(float(p) for p in parts)
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"could not parse rotations: {err}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="experiments.ood_eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Out-of-distribution (rotated-MNIST) evaluation of saved DBPCN checkpoints.",
    )
    p.add_argument(
        "--run-dirs", type=_parse_run_dirs, required=True,
        help="Comma-separated checkpoint directories. Each must contain config.json and weights.npz.",
    )
    p.add_argument(
        "--rotations", type=_parse_angles, default=(0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0),
        help="Comma-separated rotation angles in degrees. Include 0 to keep an ID baseline as a self-test.",
    )
    p.add_argument(
        "--n-test", type=int, default=10000,
        help="Max MNIST test examples loaded before class filter.",
    )
    p.add_argument(
        "--n-train", type=int, default=60000,
        help="MNIST train cap (only used to position the test slice).",
    )
    p.add_argument(
        "--mc-samples", type=int, default=16,
        help="S for the MC predictive (Eq. 34). Matches default eval convention.",
    )
    p.add_argument(
        "--eval-key", type=int, default=123,
        help="PRNG seed for the MC predictive sampling. Reused across all (run, angle) pairs for reproducibility.",
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for the eval E-step + predictive pass.",
    )
    p.add_argument(
        "--ece-bins", type=int, default=15,
        help="Number of confidence bins for ECE (standard binning).",
    )
    p.add_argument(
        "--out-dir", type=str, default="reports/ood_rotated_mnist",
        help="Output directory for the unified comparison.md / comparison.json.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _atomic_json_dump(path: str, obj) -> None:
    """Write JSON via os.replace so an interrupted run does not corrupt prior files."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def _ece(probs: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error with equal-width confidence bins.

    Standard formulation: partition examples by max-class softmax probability
    (the model's confidence), then sum |acc(bin) - conf(bin)| weighted by the
    fraction of examples in each bin.
    """
    p_max = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_idx).astype(np.float32)
    edges = np.linspace(0.0, 1.0, bins + 1)
    n = len(y_idx)
    e = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        # Include the upper edge in the last bin so p_max==1.0 is counted.
        if b == bins - 1:
            mask = (p_max >= lo) & (p_max <= hi)
        else:
            mask = (p_max >= lo) & (p_max < hi)
        if mask.any():
            conf = float(p_max[mask].mean())
            acc = float(correct[mask].mean())
            e += float(mask.sum()) / n * abs(conf - acc)
    return float(e)


def _rotate_mnist_flat(x_flat: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a batch of flat MNIST images by `angle_deg` around the image center.

    Input shape: `[N, 784]` (float). Reshapes per-image to `[28, 28]`, applies
    scipy.ndimage.rotate with reshape=False (keeps 28x28) and zero padding,
    then flattens back. Angle 0 short-circuits.
    """
    if float(angle_deg) == 0.0:
        return x_flat
    # Lazy import so the script can still --help without scipy.
    from scipy.ndimage import rotate as nd_rotate  # type: ignore

    N = x_flat.shape[0]
    imgs = x_flat.reshape(N, 28, 28)
    out = np.empty_like(imgs)
    for i in range(N):
        out[i] = nd_rotate(
            imgs[i], angle_deg, reshape=False, mode="constant", cval=0.0, order=1
        )
    # MNIST pixels are normalized to [0, 1]; bilinear interpolation can overshoot
    # slightly into [-eps, 1+eps]. Clip back to the valid range so downstream
    # moment computations stay in distribution.
    return np.clip(out, 0.0, 1.0).reshape(N, 784).astype(np.float32)


def _load_run(run_dir: str) -> Tuple[BaseConfig, "Network"]:
    """Reconstruct (BaseConfig, Network) from `run_dir`.

    Backward-compatible with older `config.json` files that may be missing
    newer fields (`output_likelihood`, `output_estimator`, `lambda_y`, etc.):
    fields missing from the JSON are filled by `BaseConfig` defaults via the
    `**kwargs` constructor path, and unrecognized JSON keys are dropped.
    """
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"{cfg_path} does not exist")
    with open(cfg_path) as f:
        raw = json.load(f)
    # `classes` is serialised as a list but BaseConfig expects a tuple.
    if "classes" in raw and isinstance(raw["classes"], list):
        raw["classes"] = tuple(raw["classes"])
    known = {f.name for f in dataclasses.fields(BaseConfig)}
    filtered = {k: v for k, v in raw.items() if k in known}
    cfg = BaseConfig(**filtered)
    _, net = _load_checkpoint(cfg, run_dir)
    return cfg, net


def _eval_dataset(net, cfg: BaseConfig, x: np.ndarray, y_idx: np.ndarray,
                  key: jax.Array, mc_samples: int, batch_size: int,
                  ece_bins: int) -> Dict[str, float]:
    """Run target-free E-step + MEAN/MC predictives on `(x, y_idx)`; return metrics.

    Identical eval geometry to `bpcn.evaluation.predict.evaluate_split`: one
    target-free E-step per batch shared between MEAN and MC predictives so the
    only difference between the two is sampling-vs-mean. Adds ECE and top-1
    confidence on top of the standard metric set.
    """
    N = len(x)
    n_batches = (N + batch_size - 1) // batch_size
    # Match `evaluate_split`'s key indexing exactly: it allocates N+1 keys and
    # consumes keys[1], keys[2], ... per batch (keys[0] is unused). Mirroring
    # that convention here means angle=0 reproduces the saved final-epoch MC
    # metrics bit-identically with the same eval-key seed.
    keys = jax.random.split(key, max(n_batches, 1) + 1)

    p_mc_all: List[np.ndarray] = []
    p_mean_all: List[np.ndarray] = []
    ent_mc_chunks: List[np.ndarray] = []
    ent_mean_chunks: List[np.ndarray] = []

    for bi in range(n_batches):
        sl = slice(bi * batch_size, min((bi + 1) * batch_size, N))
        x_b = jnp.asarray(x[sl])
        frozen = _target_free_frozen(
            net, x_b,
            T_z=cfg.eval_T_z_resolved,
            eta_m=cfg.eval_eta_m_resolved,
            eta_u=cfg.eval_eta_u_resolved,
            v_init=cfg.eval_v_init_resolved,
            objective=cfg.eval_objective_resolved,
            kappa=1.0,
            gamma_hidden=cfg.gamma_hidden,
            gamma_output=cfg.gamma_output,
        )
        p_mean = _mean_predict_from_frozen(net, frozen)
        p_mc = _mc_predict_from_frozen(net, frozen, keys[bi + 1], mc_samples)
        p_mc_all.append(np.asarray(p_mc))
        p_mean_all.append(np.asarray(p_mean))
        ent_mc_chunks.append(np.asarray(predictive_entropy(p_mc)))
        ent_mean_chunks.append(np.asarray(predictive_entropy(p_mean)))

    p_mc_arr = np.concatenate(p_mc_all, axis=0)
    p_mean_arr = np.concatenate(p_mean_all, axis=0)
    ent_mc = np.concatenate(ent_mc_chunks)
    ent_mean = np.concatenate(ent_mean_chunks)

    pred_mc = p_mc_arr.argmax(axis=-1)
    pred_mean = p_mean_arr.argmax(axis=-1)
    idx = np.arange(N)
    log_p_mc = np.log(np.clip(p_mc_arr[idx, y_idx], 1e-12, 1.0))
    log_p_mean = np.log(np.clip(p_mean_arr[idx, y_idx], 1e-12, 1.0))
    conf_mc = p_mc_arr.max(axis=-1).mean()
    conf_mean = p_mean_arr.max(axis=-1).mean()

    return {
        # MC predictive
        "MC_accuracy": float((pred_mc == y_idx).mean()),
        "MC_nll": float(-log_p_mc.mean()),
        "MC_entropy_mean": float(ent_mc.mean()),
        "MC_entropy_std": float(ent_mc.std()),
        "MC_confidence_mean": float(conf_mc),
        "MC_ece": _ece(p_mc_arr, y_idx, bins=ece_bins),
        # MEAN predictive
        "MEAN_accuracy": float((pred_mean == y_idx).mean()),
        "MEAN_nll": float(-log_p_mean.mean()),
        "MEAN_entropy_mean": float(ent_mean.mean()),
        "MEAN_entropy_std": float(ent_mean.std()),
        "MEAN_confidence_mean": float(conf_mean),
        "MEAN_ece": _ece(p_mean_arr, y_idx, bins=ece_bins),
        # Epistemic-entropy gap (Eq. 105-style proxy)
        "H_gap_mean": float((ent_mc - ent_mean).mean()),
        "H_gap_std": float((ent_mc - ent_mean).std()),
        "n": int(N),
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _markdown_table(rows: List[Dict[str, str]], cols: List[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(r.get(c, "") for c in cols) + " |")
    return "\n".join(lines)


def _write_comparison(out_dir: str, runs: List[str], angles: Tuple[float, ...],
                      results: Dict[str, Dict[float, Dict[str, float]]],
                      n_test: int, mc_samples: int, eval_key: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # comparison.json -- exact structured data
    json_payload = {
        "runs": runs,
        "angles": list(angles),
        "n_test": n_test,
        "mc_samples": mc_samples,
        "eval_key": eval_key,
        "results": {
            run: {f"{a:g}": results[run][a] for a in angles}
            for run in runs
        },
    }
    _atomic_json_dump(os.path.join(out_dir, "comparison.json"), json_payload)

    # comparison.md -- one section per metric, rows=runs, cols=angles
    def _section(title: str, key: str) -> str:
        cols = ["run"] + [f"{a:g}°" for a in angles]
        rows = []
        for run in runs:
            row = {"run": run}
            for a in angles:
                row[f"{a:g}°"] = _fmt(results[run][a].get(key, float("nan")))
            rows.append(row)
        return f"## {title}\n\n{_markdown_table(rows, cols)}\n"

    parts = [
        "# Rotated-MNIST OOD evaluation",
        "",
        f"Checkpoints: {', '.join(runs)}",
        f"Rotation angles: {', '.join(f'{a:g}°' for a in angles)}",
        f"n_test = {n_test}; mc_samples = {mc_samples}; eval_key = {eval_key}",
        "",
        "Angle 0° is the in-distribution baseline (should reproduce each run's "
        "saved final-epoch metrics within rounding). All non-zero angles are "
        "per-image rotations around the image centre with zero padding "
        "(`scipy.ndimage.rotate(reshape=False)`).",
        "",
        _section("MC accuracy",         "MC_accuracy"),
        _section("MEAN accuracy",       "MEAN_accuracy"),
        _section("MC NLL (nats)",       "MC_nll"),
        _section("MEAN NLL (nats)",     "MEAN_nll"),
        _section("MC entropy (mean)",   "MC_entropy_mean"),
        _section("MEAN entropy (mean)", "MEAN_entropy_mean"),
        _section("H_gap = H_MC − H_MEAN (mean)", "H_gap_mean"),
        _section("MC top-1 confidence", "MC_confidence_mean"),
        _section("MEAN top-1 confidence", "MEAN_confidence_mean"),
        _section("MC ECE",              "MC_ece"),
        _section("MEAN ECE",            "MEAN_ece"),
    ]
    with open(os.path.join(out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    runs = list(args.run_dirs)
    angles = tuple(float(a) for a in args.rotations)

    print(f"[ood_eval] runs={runs}")
    print(f"[ood_eval] angles={list(angles)} deg")
    print(f"[ood_eval] n_test={args.n_test}, mc_samples={args.mc_samples}, "
          f"eval_key={args.eval_key}, batch_size={args.batch_size}")

    results: Dict[str, Dict[float, Dict[str, float]]] = {}

    for run in runs:
        run_name = os.path.basename(run.rstrip(os.sep))
        print(f"\n[ood_eval] === {run_name} ===")
        t0 = time.time()
        cfg, net = _load_run(run)
        print(f"  cfg: classes={cfg.classes}, hidden_dims={cfg.hidden_dims}, "
              f"activations={cfg.activations}, output_likelihood={cfg.output_likelihood}, "
              f"output_estimator={cfg.output_estimator}")
        test = load_split(
            cfg.classes, train=False,
            seed=cfg.seed, n_train=args.n_train, n_test=args.n_test,
        )
        x_id = np.asarray(test.x)
        y_idx = np.asarray(test.y_idx)
        print(f"  loaded test split: x={x_id.shape}, n={len(y_idx)}")

        results[run_name] = {}
        for angle in angles:
            t_a = time.time()
            x_rot = _rotate_mnist_flat(x_id, angle)
            metrics = _eval_dataset(
                net, cfg, x_rot, y_idx,
                key=jax.random.PRNGKey(int(args.eval_key)),
                mc_samples=int(args.mc_samples),
                batch_size=int(args.batch_size),
                ece_bins=int(args.ece_bins),
            )
            results[run_name][angle] = metrics
            print(
                f"  angle={angle:>5g}°  "
                f"MC acc={metrics['MC_accuracy']:.4f}  MEAN acc={metrics['MEAN_accuracy']:.4f}  "
                f"MC NLL={metrics['MC_nll']:.4f}  MC H={metrics['MC_entropy_mean']:.3f}  "
                f"MEAN H={metrics['MEAN_entropy_mean']:.3f}  H_gap={metrics['H_gap_mean']:+.3f}  "
                f"({time.time() - t_a:.1f}s)"
            )

        # Per-checkpoint JSON
        per_run = {
            "run": run_name,
            "n_test": int(metrics["n"]),
            "mc_samples": int(args.mc_samples),
            "eval_key": int(args.eval_key),
            "ece_bins": int(args.ece_bins),
            "results": [
                {"angle": float(a), **results[run_name][a]} for a in angles
            ],
        }
        out_path = os.path.join(run, "ood_eval.json")
        _atomic_json_dump(out_path, per_run)
        print(f"  wrote {out_path}  ({time.time() - t0:.1f}s total for this run)")

    # Unified comparison report
    _write_comparison(
        args.out_dir, [os.path.basename(r.rstrip(os.sep)) for r in runs], angles,
        results, n_test=args.n_test, mc_samples=int(args.mc_samples),
        eval_key=int(args.eval_key),
    )
    print(f"\n[ood_eval] DONE. Comparison written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
