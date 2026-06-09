"""Per-subset, per-iter E-step residual diagnostic.

Background
----------
The DBPCN depth-fixed-point hypothesis (docs/DEPTH_FIXED_POINT_HYPOTHESIS.md)
claims the hidden variance-residual mechanism is dormant: across training
runs, `summary['hidden_l/kl_data']` reads ~10^-9 and hidden sigma^2 stays
at initialization. But that's a **batch-averaged** reading. If the
mechanism is firing on hard examples (the 20% that the model gets wrong
or is unsure about) and merely washed out by the easy 80%, the batch
mean would still read ~0. We have no evidence either way until we look
at the per-example breakdown.

This script does that lookup. No retraining, no perturbation, no new
E-step solver. It just runs the existing E-step on a saved checkpoint,
splits the test set by classification outcome, and reports per-layer
`mean(|e^l|)`, `mean(|r^l|)`, `r_pos_frac` separately on each subset and
at each of t = 0, 1, T_z/2, T_z latent-descent iterations.

Outputs
-------
- `<out-dir>/trace.json` — nested
  `{subset: {layer_l: {iter_t: {e_abs, r_abs, r_pos_frac}}}}`.
- `<out-dir>/summary.md` — three subset tables plus a decision-quadrant
  resolution against the four hypotheses in
  `docs/DEPTH_FIXED_POINT_HYPOTHESIS.md`.

References
----------
- v2 Eqs. 60-61: `moment_forward` for `(m_p, v_p)`.
- v2 Eqs. 66, 68: `e = m_z - m_p`, `r = v_z + e^2 - v_p`.
- v2 Eq. 89 + extension Eq. 22-23: predictive latent init + shared-energy E-step.
- continuation Eq. 26: psi_L on the top latent.

Usage
-----
    python -m experiments.hard_subset_residuals \\
        --run-dir runs/cat_mc_10c2_rerun2 \\
        --out-dir reports/l3_hard_subset \\
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from bpcn.configs.base import BaseConfig
from bpcn.data.mnist import load_split
from bpcn.evaluation.predict import (
    _mc_predict_from_frozen,
    _target_free_frozen,
)
from bpcn.inference.e_step import e_step
from bpcn.inference.feature_moments import psi_moments
from bpcn.models.moments import moment_forward

# Reuse the canonical loader from ood_eval.py — it already handles the
# field-filtering for unknown keys in old config.json files.
from experiments.ood_eval import _load_run


_SUBSET_OFFSETS = {"clean": 11, "confused": 23, "wrong": 37}


def _subset_offset(name: str) -> int:
    try:
        return _SUBSET_OFFSETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown subset name: {name!r}") from exc


# ---------------------------------------------------------------------------
# Subset masks
# ---------------------------------------------------------------------------

def _build_subset_masks(
    p_mc: np.ndarray, y_idx: np.ndarray, *, conf_clean: float, conf_confused: float,
) -> Dict[str, np.ndarray]:
    """Three non-overlapping subset masks based on MC predictions.

    - `clean`:    correct AND max(p_mc) >= conf_clean
    - `confused`: correct AND max(p_mc) <= conf_confused
    - `wrong`:    argmax(p_mc) != y
    """
    pred = p_mc.argmax(axis=1)
    pmax = p_mc.max(axis=1)
    correct = pred == y_idx
    masks = {
        "clean": correct & (pmax >= conf_clean),
        "confused": correct & (pmax <= conf_confused),
        "wrong": ~correct,
    }
    return masks


# ---------------------------------------------------------------------------
# Per-iter residual computation
# ---------------------------------------------------------------------------

def _compute_residuals(
    net, x: jax.Array, m_zs: Tuple[jax.Array, ...], v_zs: Tuple[jax.Array, ...],
) -> Dict[int, Dict[str, float]]:
    """For each hidden layer l, compute mean|e|, mean|r|, r_pos_frac.

    Uses the canonical formulas from `bpcn.training.m_step.update_layer`
    lines 106-107:
        e = m_z - m_p
        r = v_z + e^2 - v_p
    where (m_p, v_p) are the predictive moments via `moment_forward` and
    the presynaptic moments come from `psi_moments` (interior) or the
    raw input (layer 0).
    """
    out: Dict[int, Dict[str, float]] = {}
    for l in range(net.L_hidden):
        if l == 0:
            M, V = x, jnp.zeros_like(x)
        else:
            M, V = psi_moments(net.activations[l - 1], m_zs[l - 1], v_zs[l - 1])
        m_p, v_p = moment_forward(net.layers[l], M, V)
        e = m_zs[l] - m_p
        r = v_zs[l] + e * e - v_p
        out[l] = {
            "e_abs": float(jnp.abs(e).mean()),
            "r_abs": float(jnp.abs(r).mean()),
            "r_pos_frac": float((r > 0).mean()),
        }
    return out


def _e_step_with_trace(net, cfg: BaseConfig, x: jax.Array, y_idx: jax.Array,
                       key: jax.Array, S_train: int):
    """Run the actual training-objective E-step on (x, y_idx).

    Descends the real F_DPC with the actual labels (not target-free) so
    F_out is active. This keeps the measurement faithful to what the
    M-step sees during training. record_per_iter=True returns the
    intermediate (m_t, u_t) trace.

    Note: we set output_weight to cfg.lambda_y so the E-step descent is
    weighted exactly as during training. Gaussian-mode runs get
    output_weight=lambda_y as well — y_mean / y_var act as the target;
    we pass zero placeholders since the categorical path actually used by
    the runs of interest consumes only y_idx (categorical_output.py).
    """
    B = x.shape[0]
    y_mean = jnp.zeros((B, cfg.output_dim), dtype=x.dtype)
    y_var = jnp.zeros_like(y_mean)
    frozen, diag = e_step(
        net, x, y_mean,
        T_z=cfg.eval_T_z_resolved,
        eta_m=cfg.eval_eta_m_resolved,
        eta_u=cfg.eval_eta_u_resolved,
        v_init=cfg.eval_v_init_resolved,
        output_weight=cfg.lambda_y,
        y_var=y_var,
        gamma_hidden=cfg.gamma_hidden,
        gamma_output=cfg.gamma_output,
        y_idx=y_idx,
        key=key,
        mc_samples_train=S_train,
        record_per_iter=True,
    )
    return frozen, diag


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _select_iters(T_z: int):
    """Iteration snapshots from the proposal: t = 0, 1, T_z/2, T_z."""
    # 0 = initial (pre-step); 1 = after iter 1; mid = after iter T_z/2;
    # final = after iter T_z (frozen).
    return [0, 1, max(1, T_z // 2), T_z]


def _gather_subset(arr: np.ndarray, mask: np.ndarray):
    """Numpy → jax for the masked subset."""
    return jnp.asarray(arr[mask])


def run_diagnostic(
    run_dir: str,
    out_dir: str,
    *,
    seed: int = 0,
    mc_samples_pred: int = 16,
    conf_clean: float = 0.8,
    conf_confused: float = 0.4,
    max_subset: int = 2000,
):
    """Main entry point. Produces trace.json + summary.md under `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"[hard-subset] Loading checkpoint from {run_dir} ...")
    cfg, net = _load_run(run_dir)
    print(f"  hidden_dims={cfg.hidden_dims} activations={cfg.activations}")
    print(f"  output_likelihood={cfg.output_likelihood} estimator={cfg.output_estimator}")
    print(f"  T_z(eval)={cfg.eval_T_z_resolved} S_train={cfg.mc_samples_train}")

    # Load the test split for the saved class set.
    print(f"[hard-subset] Loading MNIST test split for classes {cfg.classes} ...")
    test = load_split(cfg.classes, train=False, seed=cfg.seed)
    print(f"  n_test = {len(test.x)}")

    # --- MC predictive on full test set to build subset masks ---
    print(f"[hard-subset] Running target-free MC predictive (S={mc_samples_pred}) ...")
    key = jax.random.PRNGKey(seed)
    key, k_pred_e, k_pred_mc = jax.random.split(key, 3)

    # We need per-example p_mc. Use the same target-free E-step + MC
    # predictive infrastructure that evaluate_split uses, but in batches
    # so we can scale to the full 10k test set.
    p_mc_chunks = []
    batch_pred = 512
    N = len(test.x)
    for i in range(0, N, batch_pred):
        sl = slice(i, min(i + batch_pred, N))
        x_b = jnp.asarray(test.x[sl])
        frozen_b = _target_free_frozen(
            net, x_b,
            T_z=cfg.eval_T_z_resolved,
            eta_m=cfg.eval_eta_m_resolved,
            eta_u=cfg.eval_eta_u_resolved,
            v_init=cfg.eval_v_init_resolved,
            gamma_hidden=cfg.gamma_hidden,
            gamma_output=cfg.gamma_output,
        )
        kb = jax.random.fold_in(k_pred_mc, i)
        p_mc_b = _mc_predict_from_frozen(net, frozen_b, kb, mc_samples_pred)
        p_mc_chunks.append(np.asarray(p_mc_b))
    p_mc = np.concatenate(p_mc_chunks, axis=0)
    print(f"  MC predictive shape: {p_mc.shape}")

    # --- Build subset masks ---
    masks = _build_subset_masks(p_mc, test.y_idx, conf_clean=conf_clean,
                                conf_confused=conf_confused)
    for name, mask in masks.items():
        print(f"  subset '{name}': {mask.sum()} examples ({100*mask.mean():.1f}%)")

    # --- Per-subset E-step with per-iter trace ---
    T_z = cfg.eval_T_z_resolved
    iter_snapshots = _select_iters(T_z)
    print(f"[hard-subset] Iteration snapshots: t = {iter_snapshots}")

    trace: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    subset_meta: Dict[str, Dict[str, float]] = {}
    for name, mask in masks.items():
        n_avail = int(mask.sum())
        if n_avail == 0:
            print(f"[hard-subset] Subset '{name}' is empty; skipping.")
            continue
        n_use = min(n_avail, max_subset)
        # Deterministic subsample if the subset is too large for one E-step.
        if n_use < n_avail:
            rng = np.random.default_rng(seed + _subset_offset(name))
            idx = rng.choice(np.where(mask)[0], size=n_use, replace=False)
        else:
            idx = np.where(mask)[0]
        x_sub = jnp.asarray(test.x[idx])
        y_idx_sub = jnp.asarray(test.y_idx[idx], dtype=jnp.int32)
        subset_meta[name] = {
            "n_avail": float(n_avail),
            "n_used": float(n_use),
            "frac_of_test": float(n_avail) / float(N),
        }

        print(f"[hard-subset] E-step on '{name}' (n={n_use}, T_z={T_z}) ...")
        k_sub = jax.random.fold_in(key, _subset_offset(name))
        frozen_sub, diag_sub = _e_step_with_trace(
            net, cfg, x_sub, y_idx_sub, k_sub, S_train=cfg.mc_samples_train,
        )

        # Per-iter residuals. iter index 0 = initial latents (m_0, u_0).
        # iter t in [1, T_z] = after t scan steps -> m_trace[t-1], u_trace[t-1].
        per_iter: Dict[str, Dict[str, Dict[str, float]]] = {}
        for t in iter_snapshots:
            if t == 0:
                m_t = diag_sub.m_initial_trace
                u_t = diag_sub.u_initial_trace
            else:
                # scan index t-1
                m_t = tuple(arr[t - 1] for arr in diag_sub.m_trace)
                u_t = tuple(arr[t - 1] for arr in diag_sub.u_trace)
            v_t = tuple(jnp.exp(u) for u in u_t)
            r = _compute_residuals(net, x_sub, m_t, v_t)
            per_iter[f"t{t}"] = {f"layer_{l}": vals for l, vals in r.items()}
        trace[name] = per_iter

        # Also report the F_drop and final F for this subset.
        F_drop = float(diag_sub.F_initial - diag_sub.F_final)
        subset_meta[name].update({
            "F_initial": float(diag_sub.F_initial),
            "F_final": float(diag_sub.F_final),
            "F_drop": F_drop,
        })

    # --- Write trace.json ---
    json_path = os.path.join(out_dir, "trace.json")
    payload = {
        "run_dir": run_dir,
        "config": {
            "hidden_dims": list(cfg.hidden_dims),
            "activations": list(cfg.activations),
            "output_likelihood": cfg.output_likelihood,
            "output_estimator": cfg.output_estimator,
            "T_z": T_z,
            "mc_samples_train": cfg.mc_samples_train,
            "lambda_y": cfg.lambda_y,
        },
        "subset_thresholds": {
            "conf_clean": conf_clean,
            "conf_confused": conf_confused,
            "mc_samples_pred": mc_samples_pred,
        },
        "subset_meta": subset_meta,
        "iter_snapshots": iter_snapshots,
        "trace": trace,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[hard-subset] Wrote {json_path}")

    # --- Write summary.md ---
    md_path = os.path.join(out_dir, "summary.md")
    _write_summary_md(md_path, payload, net.L_hidden, iter_snapshots)
    print(f"[hard-subset] Wrote {md_path}")


def _write_summary_md(path: str, payload: dict, L_hidden: int, iters):
    """Render a markdown summary with one table per subset + decision quadrant."""
    cfg = payload["config"]
    sm = payload["subset_meta"]
    trace = payload["trace"]

    lines = []
    lines.append(f"# Hard-subset residual diagnostic — {payload['run_dir']}")
    lines.append("")
    lines.append("Diagnostic for whether the hidden DBPCN variance-residual mechanism")
    lines.append("is genuinely dormant or merely washed out by batch averaging.")
    lines.append("Stratifies the MNIST test set by MC-prediction outcome and reports")
    lines.append("per-layer `mean(|e^l|)`, `mean(|r^l|)`, `r_pos_frac` at iterations")
    lines.append(f"t = {iters} of the E-step (T_z = {cfg['T_z']}).")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- hidden_dims: `{cfg['hidden_dims']}`")
    lines.append(f"- activations: `{cfg['activations']}`")
    lines.append(f"- output: `{cfg['output_likelihood']}` / `{cfg['output_estimator']}`")
    lines.append(f"- T_z: {cfg['T_z']}  mc_samples_train: {cfg['mc_samples_train']}  lambda_y: {cfg['lambda_y']}")
    thr = payload["subset_thresholds"]
    lines.append(f"- subset thresholds: clean >= {thr['conf_clean']}, confused <= {thr['conf_confused']} (MC top-1 conf)")
    lines.append(f"- MC samples for subset prediction: {thr['mc_samples_pred']}")
    lines.append("")
    lines.append("## Subset sizes and F-drop per subset")
    lines.append("")
    lines.append("| subset | n_avail | n_used | frac of test | F_initial | F_final | F_drop |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ("clean", "confused", "wrong"):
        if name not in sm:
            continue
        m = sm[name]
        lines.append(
            f"| {name} | {int(m['n_avail'])} | {int(m['n_used'])} | {m['frac_of_test']:.3f} "
            f"| {m['F_initial']:.4f} | {m['F_final']:.4f} | {m['F_drop']:.4e} |"
        )
    lines.append("")
    lines.append("## Per-subset per-layer per-iter residuals")
    lines.append("")

    iter_keys = [f"t{t}" for t in iters]

    for name in ("clean", "confused", "wrong"):
        if name not in trace:
            continue
        lines.append(f"### Subset: `{name}`  (n = {int(sm[name]['n_used'])})")
        lines.append("")
        for metric_key, metric_label in [
            ("e_abs", "mean(|e^l|)"),
            ("r_abs", "mean(|r^l|)"),
            ("r_pos_frac", "r_pos_frac"),
        ]:
            header = "| layer | " + " | ".join(f"t={t}" for t in iters) + " |"
            divider = "|---|" + "|".join("---:" for _ in iters) + "|"
            lines.append(f"**{metric_label}**")
            lines.append("")
            lines.append(header)
            lines.append(divider)
            for l in range(L_hidden):
                row = [f"hidden_{l}"]
                for it in iter_keys:
                    v = trace[name][it][f"layer_{l}"][metric_key]
                    if metric_key == "r_pos_frac":
                        row.append(f"{v:.4f}")
                    else:
                        row.append(f"{v:.3e}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # Decision quadrant — populated heuristically from the trace.
    lines.append("## Decision quadrant")
    lines.append("")
    lines.append("Reading the trace against the four hypotheses from")
    lines.append("`docs/DEPTH_FIXED_POINT_HYPOTHESIS.md`:")
    lines.append("")
    # Heuristic verdict on Diag 1: is hard-subset e_abs > 100x easy-subset e_abs?
    diag1, diag2 = None, None
    if "clean" in trace and "wrong" in trace:
        # Compare e_abs at final iter, layer 0 (the deepest from output)
        final_key = iter_keys[-1]
        # Find max ratio across layers (sometimes interior layer has different signal)
        ratios = []
        for l in range(L_hidden):
            e_clean = trace["clean"][final_key][f"layer_{l}"]["e_abs"]
            e_wrong = trace["wrong"][final_key][f"layer_{l}"]["e_abs"]
            if e_clean > 0:
                ratios.append(e_wrong / max(e_clean, 1e-12))
        max_ratio = max(ratios) if ratios else 0.0
        diag1 = "hard >> easy" if max_ratio > 10.0 else "small everywhere"
        lines.append(f"- **Diag 1 (hard-subset `|e^l|` per layer):** "
                     f"max ratio wrong/clean = {max_ratio:.2f} → **{diag1}**")
    if "wrong" in trace:
        # Diag 2: is the wrong-subset trace still climbing at t=T_z?
        # Use top-layer e_abs as proxy.
        ekey = iter_keys
        wrong_top = [trace["wrong"][k][f"layer_{L_hidden-1}"]["e_abs"] for k in ekey]
        if len(wrong_top) >= 3:
            # Compare last vs mid: if last > mid * 1.05, still climbing.
            still_climbing = wrong_top[-1] > wrong_top[len(wrong_top)//2] * 1.05
            # And: flat from t≈4 means last ≈ mid (within 5%).
            flat_late = abs(wrong_top[-1] - wrong_top[-2]) < 0.05 * max(wrong_top[-1], 1e-12)
            diag2 = ("still climbing at t=T_z" if still_climbing
                     else "flat from mid-iter onward" if flat_late
                     else "ambiguous")
            lines.append(f"- **Diag 2 (`|e^l|` trace on `wrong` subset, top layer):** "
                         f"{wrong_top} → **{diag2}**")
    lines.append("")
    lines.append("**Mapped to next-experiment priority:**")
    lines.append("")
    quad = (diag1 or "?", diag2 or "?")
    verdict = {
        ("small everywhere", "flat from mid-iter onward"):
            "Dormancy confirmed at per-example resolution. Perturbed-init experiment is the only path forward.",
        ("hard >> easy", "flat from mid-iter onward"):
            "Mechanism is firing on hard examples but invisible in batch mean. Perturbed init becomes lower priority; investigate hardness-weighted M-step or longer training on hard examples only.",
        ("small everywhere", "still climbing at t=T_z"):
            "E-step undertrained. Bump T_z to 32-48 before designing new init.",
        ("hard >> easy", "still climbing at t=T_z"):
            "Both effects present. Higher inner budget (T_z) first, then evaluate init.",
    }.get(quad, "Outcome is ambiguous; inspect the per-iter tables manually.")
    lines.append(f"> Quadrant `{quad}` → {verdict}")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="experiments.hard_subset_residuals",
        description="Per-subset, per-iter E-step residual diagnostic for DBPCN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=str, required=True,
                   help="Path to a saved run dir with config.json + weights.npz.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory under reports/ for trace.json + summary.md.")
    p.add_argument("--seed", type=int, default=0,
                   help="PRNG seed for MC predictives and subset subsampling.")
    p.add_argument("--mc-samples-pred", type=int, default=16,
                   help="S for the MC predictive used to build subset masks.")
    p.add_argument("--conf-clean", type=float, default=0.8,
                   help="MC top-1 confidence threshold for the 'clean' bucket.")
    p.add_argument("--conf-confused", type=float, default=0.4,
                   help="MC top-1 confidence threshold for the 'confused' bucket.")
    p.add_argument("--max-subset", type=int, default=2000,
                   help="If a subset is bigger than this, subsample for tractability.")
    args = p.parse_args(argv)

    run_diagnostic(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        mc_samples_pred=args.mc_samples_pred,
        conf_clean=args.conf_clean,
        conf_confused=args.conf_confused,
        max_subset=args.max_subset,
    )


if __name__ == "__main__":
    main()
