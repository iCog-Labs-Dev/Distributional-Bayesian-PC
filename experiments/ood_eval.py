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

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from bpcn.configs.base import BaseConfig
from bpcn.data.mnist import load_split
from bpcn.evaluation.predict import _eval_one_batch
from experiments.mnist import _load_checkpoint


@partial(jax.jit, static_argnames=(
    "n_batches", "batch_size", "T_z", "mc_samples",
))
def _scan_predict_padded_jit(
    net, x_padded, keys, *,
    n_batches, batch_size,
    T_z, eta_m, eta_u, v_init,
    gamma_hidden, gamma_output, mc_samples,
):
    """Scan `_eval_one_batch` over [n_batches, batch_size, ...] and return
    the concatenated MC + MEAN probability arrays.

    Shared OOD eval primitive: identical math to the per-batch Python loop
    it replaces, but in one fused XLA graph so the only device->host sync
    is the final `np.asarray` call after this returns.
    """
    x_b = x_padded.reshape(n_batches, batch_size, -1)

    def step(_carry, args):
        k, xb = args
        p_mc, p_mean = _eval_one_batch(
            net, xb, k,
            T_z=T_z, eta_m=eta_m, eta_u=eta_u, v_init=v_init,
            gamma_hidden=gamma_hidden, gamma_output=gamma_output,
            mc_samples=mc_samples,
        )
        return _carry, (p_mc, p_mean)

    _, (p_mc_b, p_mean_b) = jax.lax.scan(step, None, (keys, x_b))
    # [n_batches, batch_size, C] -> [N_padded, C]
    return (
        p_mc_b.reshape(n_batches * batch_size, -1),
        p_mean_b.reshape(n_batches * batch_size, -1),
    )


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
    p.add_argument(
        "--with-layer-residuals", action="store_true",
        help="Also compute per-hidden-layer wrong-subset residuals (K, |e|, |r|, "
             "r_pos_frac) at each angle, using the seed-free target-free E-step "
             "(init_perturb_std=0.0). The wrong subset is defined by MC argmax "
             "!= y_idx on the rotated inputs (continuation Section 6.6). "
             "Decisive diagnostic for the perturbed-init confirmation experiment.")
    p.add_argument(
        "--wrong-subset-max", type=int, default=None,
        help="Cap on the number of wrong-subset examples used for the per-layer "
             "residual computation when --with-layer-residuals is set. "
             "Default = --batch-size (single fixed-shape trace per angle; "
             "avoids JIT retracing across angles).")
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
                  ece_bins: int,
                  return_arrays: bool = False):
    """Run target-free E-step + MEAN/MC predictives on `(x, y_idx)`; return metrics.

    Identical eval geometry to `bpcn.evaluation.predict.evaluate_split`: one
    target-free E-step per batch shared between MEAN and MC predictives so the
    only difference between the two is sampling-vs-mean. Adds ECE and top-1
    confidence on top of the standard metric set.

    Runs as a single fused JAX graph via `_scan_predict_padded_jit`:
    pad-to-multiple-of-batch + scan + one device->host transfer of the
    concatenated probability arrays. ECE is computed host-side via
    `_ece(probs, y_idx, bins)` because it requires the full array for bin
    assignment; everything else is a one-line numpy reduction over the
    transferred probs.
    """
    N = len(x)
    n_batches = (N + batch_size - 1) // batch_size
    N_padded = n_batches * batch_size
    pad = N_padded - N

    x_arr = jnp.asarray(x)
    if pad > 0:
        x_padded = jnp.concatenate(
            [x_arr, jnp.zeros((pad, x_arr.shape[1]), dtype=x_arr.dtype)],
            axis=0,
        )
    else:
        x_padded = x_arr

    # Match `evaluate_split`'s key-allocation convention exactly so the
    # angle=0 self-test reproduces the saved final-epoch MC metrics
    # bit-identically with the same eval-key seed: allocate `n_batches + 1`
    # keys and discard the 0th.
    keys = jax.random.split(key, max(n_batches, 1) + 1)[1:]

    p_mc_padded, p_mean_padded = _scan_predict_padded_jit(
        net, x_padded, keys,
        n_batches=n_batches, batch_size=batch_size,
        T_z=int(cfg.eval_T_z_resolved),
        eta_m=float(cfg.eval_eta_m_resolved),
        eta_u=float(cfg.eval_eta_u_resolved),
        v_init=float(cfg.eval_v_init_resolved),
        gamma_hidden=float(cfg.gamma_hidden),
        gamma_output=float(cfg.gamma_output),
        mc_samples=int(mc_samples),
    )

    # Single device->host transfer; trim padded rows.
    p_mc_arr = np.asarray(p_mc_padded)[:N]
    p_mean_arr = np.asarray(p_mean_padded)[:N]

    pred_mc = p_mc_arr.argmax(axis=-1)
    pred_mean = p_mean_arr.argmax(axis=-1)
    idx = np.arange(N)
    log_p_mc = np.log(np.clip(p_mc_arr[idx, y_idx], 1e-12, 1.0))
    log_p_mean = np.log(np.clip(p_mean_arr[idx, y_idx], 1e-12, 1.0))
    conf_mc = p_mc_arr.max(axis=-1).mean()
    conf_mean = p_mean_arr.max(axis=-1).mean()

    # Per-example entropies (Eq. 104). Computed host-side off the already-
    # transferred probability arrays so we don't pay a second device sync.
    p_mc_clip = np.clip(p_mc_arr, 1e-12, 1.0)
    p_mean_clip = np.clip(p_mean_arr, 1e-12, 1.0)
    ent_mc = -np.sum(p_mc_clip * np.log(p_mc_clip), axis=-1)
    ent_mean = -np.sum(p_mean_clip * np.log(p_mean_clip), axis=-1)

    metrics = {
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
    if return_arrays:
        return metrics, p_mc_arr
    return metrics


def _wrong_subset_layer_residuals(net, cfg: BaseConfig,
                                  x_rot: np.ndarray, p_mc_arr: np.ndarray,
                                  y_idx: np.ndarray, batch_size: int,
                                  max_n: int) -> Dict:
    """Per-hidden-layer K, |e|, |r|, r_pos_frac on the wrong subset.

    Wrong = MC argmax != y_idx, same definition as
    `experiments/hard_subset_residuals.py:_build_subset_masks` and consistent
    with continuation Section 6.6 (target-free predictive inference).
    Uses the seed-free target-free E-step via `_target_free_frozen`
    (init_perturb_std=0.0 -- locked by `bpcn/evaluation/predict.py` defensive
    arg) so the residuals reflect the trained checkpoint's intrinsic latent
    state, not the training-time perturbation.

    To avoid JIT retraces across the 8 OOD angles, the wrong subset is capped
    to `min(len(wrong), max_n)` examples and (if needed) padded to the next
    multiple of `batch_size`. Residuals are batch-averaged across the valid
    (non-pad) examples only.

    Returns
    -------
    dict[int, dict] keyed by layer index l, each value
    `{K, e_abs, r_abs, r_pos_frac, n_wrong, n_used}`.
    None if the wrong subset is empty.
    """
    # Local imports to keep ood_eval importable without these dependencies
    # under headless eval paths that don't request residuals.
    from bpcn.evaluation.predict import _target_free_frozen
    from bpcn.inference.shared_energy import per_layer_residuals as _per_layer_residuals

    pred_mc = p_mc_arr.argmax(axis=-1)
    wrong_mask = (pred_mc != y_idx)
    n_wrong = int(wrong_mask.sum())
    if n_wrong == 0:
        return None
    wrong_idx = np.where(wrong_mask)[0]
    n_used = min(n_wrong, int(max_n))
    sub_idx = wrong_idx[:n_used]
    x_w = x_rot[sub_idx]
    # Pad to a multiple of batch_size so the JIT trace is shape-stable across
    # angles when n_used is the same constant; the per_layer_residuals helper
    # returns batch-mean reductions, so we slice to the valid prefix BEFORE
    # taking moments to keep the averaging honest.
    pad = (-len(x_w)) % batch_size
    if pad > 0:
        x_padded = np.concatenate(
            [x_w, np.zeros((pad, x_w.shape[1]), dtype=x_w.dtype)],
            axis=0,
        )
    else:
        x_padded = x_w
    x_padded_j = jnp.asarray(x_padded)
    frozen = _target_free_frozen(
        net, x_padded_j,
        T_z=int(cfg.eval_T_z_resolved),
        eta_m=float(cfg.eval_eta_m_resolved),
        eta_u=float(cfg.eval_eta_u_resolved),
        v_init=float(cfg.eval_v_init_resolved),
        gamma_hidden=float(cfg.gamma_hidden),
        gamma_output=float(cfg.gamma_output),
    )
    # Slice the frozen-latent tuples back to the valid prefix so the residual
    # averages reflect only real wrong-subset examples (pad rows are dummy
    # zeros and would dilute the means).
    m_zs_valid = tuple(m[:n_used] for m in frozen.m_zs)
    v_zs_valid = tuple(v[:n_used] for v in frozen.v_zs)
    x_valid = x_padded_j[:n_used]
    residuals = _per_layer_residuals(net, x_valid, m_zs_valid, v_zs_valid)
    return {
        l: {
            "K": float(np.asarray(d["K"])),
            "e_abs": float(np.asarray(d["e_abs"])),
            "r_abs": float(np.asarray(d["r_abs"])),
            "r_pos_frac": float(np.asarray(d["r_pos_frac"])),
            "n_wrong": n_wrong,
            "n_used": n_used,
        }
        for l, d in residuals.items()
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

    # Optional per-hidden-layer wrong-subset residuals (--with-layer-residuals).
    # Layout: one section per metric (K / e_abs / r_abs / r_pos_frac), inside
    # each section one sub-table per run with rows=layer indices and cols=angles.
    # Only emit if at least one (run, angle) cell carries the residual payload.
    def _layer_keys_present():
        for run in runs:
            for a in angles:
                cell = results[run][a].get("wrong_subset_layer_residuals")
                if cell:
                    return sorted(cell.keys())
        return None

    layer_keys = _layer_keys_present()
    if layer_keys is not None:
        parts.append(
            "## Wrong-subset per-hidden-layer residuals\n\n"
            "Per-hidden-layer K (mean inclusion-KL), |e|, |r|, r_pos_frac on the "
            "MC-wrong subset (pred_mc ≠ y_idx), evaluated at the seed-free "
            "target-free E-step fixed point (init_perturb_std=0.0 — locked by "
            "`bpcn/evaluation/predict.py`). Decisive for whether the σ² that "
            "moved during training carries data-conditional epistemic mass "
            "rather than just tracking the training-time perturbation noise.\n"
        )

        def _residual_section(title: str, key: str) -> str:
            cols = ["run", "layer"] + [f"{a:g}°" for a in angles]
            rows = []
            for run in runs:
                for lk in layer_keys:
                    row = {"run": run, "layer": lk}
                    for a in angles:
                        cell = results[run][a].get("wrong_subset_layer_residuals")
                        if cell and lk in cell:
                            row[f"{a:g}°"] = _fmt(cell[lk].get(key, float("nan")))
                        else:
                            row[f"{a:g}°"] = ""
                    rows.append(row)
            return f"### {title}\n\n{_markdown_table(rows, cols)}\n"

        parts.extend([
            _residual_section("K (mean per-unit inclusion-KL)", "K"),
            _residual_section("|e| (mean abs mean-error)",       "e_abs"),
            _residual_section("|r| (mean abs variance residual)", "r_abs"),
            _residual_section("r_pos_frac",                      "r_pos_frac"),
        ])

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
        wrong_subset_max = (
            int(args.wrong_subset_max)
            if args.wrong_subset_max is not None
            else int(args.batch_size)
        )
        for angle in angles:
            t_a = time.time()
            x_rot = _rotate_mnist_flat(x_id, angle)
            eval_out = _eval_dataset(
                net, cfg, x_rot, y_idx,
                key=jax.random.PRNGKey(int(args.eval_key)),
                mc_samples=int(args.mc_samples),
                batch_size=int(args.batch_size),
                ece_bins=int(args.ece_bins),
                return_arrays=bool(args.with_layer_residuals),
            )
            if args.with_layer_residuals:
                metrics, p_mc_arr = eval_out
                wrong_res = _wrong_subset_layer_residuals(
                    net, cfg, x_rot, p_mc_arr, y_idx,
                    batch_size=int(args.batch_size),
                    max_n=wrong_subset_max,
                )
                # Stored as {"layer_l": {...}} so JSON serialisation handles
                # string keys cleanly without int-key coercion at load time.
                metrics["wrong_subset_layer_residuals"] = (
                    {f"layer_{l}": d for l, d in wrong_res.items()}
                    if wrong_res is not None else None
                )
            else:
                metrics = eval_out
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
