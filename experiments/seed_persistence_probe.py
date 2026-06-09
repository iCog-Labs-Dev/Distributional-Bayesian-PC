"""Seed-persistence trajectory probe.

Background
----------
The injection-coupling probe (`reports/l3_injection_probe/`, World B +
delta-OK verdict) proved a single 0.1·σ kick at an interior latent
transmits in both directions at coupling magnitudes comparable to the
top-layer reference. The coupling is alive *at the moment of injection*.

But the M-step never sees `t = 0`. It sees the **frozen** latents at
`t = T_z`, after the E-step has had T_z gradient steps to descend
`F_DPC`. And the E-step's descent objective includes the per-layer
transition KL `K^l = KL(N(m_z^l, v_z^l) ‖ N(m_p^l, v_p^l))`, which is
*strictly larger* than zero whenever `e^l ≠ 0`. So the E-step has an
active gradient incentive to erase the very seed a perturbed-init
training run would inject.

This script tests whether the seed actually survives. Same checkpoint,
same E-step, same examples (the `wrong` subset). Seed *all* hidden
layers simultaneously with `m_z^l ← m_p^l + 0.1·sqrt(v_p^l)·ξ`, run the
ordinary E-step with `record_per_iter=True`, and log per-layer `|e^l|`,
`|r^l|`, `r_pos_frac` at `t = 0, 1, T_z/2, T_z`. Subtract a
matched-pair baseline (the same E-step with `init_perturb_std=0.0`) to
isolate the seed's contribution from the top-layer's natural ~10⁻⁵
drift.

Decision matrix
---------------
For each hidden layer:
    ρ^l = |e^l|_delta(t=T_z) / |e^l|_delta(t=0)
where |e^l|_delta = |e^l|_seeded − |e^l|_baseline.

Aggregate `ρ̄ = min_l ρ^l` over interior layers (l < L_hidden - 1).

- `ρ̄ ≥ 0.1`     → seed survives within 1 OOM → **green light** for
                   perturbed-init training run.
- `0.01 ≤ ρ̄ < 0.1` → partial decay → **yellow**: raise δ or switch to
                       per-E-step re-seed.
- `ρ̄ < 0.01`    → full decay → **redirect** to per-E-step re-seed
                   before any training run.
- `ρ̄ > 1.0`     → seed grows → **instability discovery**.

Plus the F_DPC drop comparison (cross-check):
- seeded ΔF >> baseline ΔF → active erasure pressure on the seed.
- seeded ΔF ≈ baseline ΔF  → seed is orthogonal to descent (or descent
                              has nothing to do at predictive init).

References
----------
- v2 Eqs. 60-61: `moment_forward` for (m_p, v_p).
- v2 Eq. 89:     `initial_latents` predictive feedforward.
- v2 Eqs. 66, 68: `e = m_z − m_p`, `r = v_z + e² − v_p`.
- Extension Eq. 12: F_DPC, the shared free energy the E-step descends.

Usage
-----
    python -m experiments.seed_persistence_probe \\
        --run-dir runs/cat_mc_10c2_rerun2 \\
        --out-dir reports/l3_seed_persistence \\
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

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
from experiments.hard_subset_residuals import (
    _build_subset_masks,
    _compute_residuals,
    _subset_offset,
)
from experiments.ood_eval import _load_run


# ---------------------------------------------------------------------------
# Subset construction (reuses hard_subset_residuals masks for cross-comparison)
# ---------------------------------------------------------------------------

def _build_p_mc(net, cfg: BaseConfig, x: np.ndarray, key: jax.Array,
                mc_samples: int, batch_pred: int = 512) -> np.ndarray:
    """Per-example MC predictive over a test split."""
    p_chunks = []
    N = len(x)
    for i in range(0, N, batch_pred):
        sl = slice(i, min(i + batch_pred, N))
        x_b = jnp.asarray(x[sl])
        frozen_b = _target_free_frozen(
            net, x_b,
            T_z=cfg.eval_T_z_resolved,
            eta_m=cfg.eval_eta_m_resolved,
            eta_u=cfg.eval_eta_u_resolved,
            v_init=cfg.eval_v_init_resolved,
            gamma_hidden=cfg.gamma_hidden,
            gamma_output=cfg.gamma_output,
        )
        kb = jax.random.fold_in(key, i)
        p_mc_b = _mc_predict_from_frozen(net, frozen_b, kb, mc_samples)
        p_chunks.append(np.asarray(p_mc_b))
    return np.concatenate(p_chunks, axis=0)


# ---------------------------------------------------------------------------
# E-step with per-iter trace, optionally seeded
# ---------------------------------------------------------------------------

def _e_step_traced(net, cfg: BaseConfig, x: jax.Array, y_idx: jax.Array,
                   key: jax.Array, *, perturb_std: float, perturb_key):
    """Run the training-objective E-step (real labels, λ_y, real T_z) with
    `record_per_iter=True`. Returns (frozen, diag).

    Mirrors the call pattern from
    `experiments/hard_subset_residuals.py::_e_step_with_trace` but adds
    the `init_perturb_std` / `init_perturb_key` kwargs.
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
        mc_samples_train=cfg.mc_samples_train,
        record_per_iter=True,
        init_perturb_std=perturb_std,
        init_perturb_key=perturb_key,
    )
    return frozen, diag


# ---------------------------------------------------------------------------
# Per-iter residual collection across the trace
# ---------------------------------------------------------------------------

def _collect_per_iter_residuals(
    net, x: jax.Array, diag, iter_snapshots: List[int],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """For each t in iter_snapshots, reconstruct (m_zs, v_zs) and call
    `_compute_residuals`. Returns
    `{"t{t}": {"layer_l": {e_abs, r_abs, r_pos_frac}}}`.
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for t in iter_snapshots:
        if t == 0:
            m_t = diag.m_initial_trace
            u_t = diag.u_initial_trace
        else:
            m_t = tuple(arr[t - 1] for arr in diag.m_trace)
            u_t = tuple(arr[t - 1] for arr in diag.u_trace)
        v_t = tuple(jnp.exp(u) for u in u_t)
        r = _compute_residuals(net, x, m_t, v_t)
        out[f"t{t}"] = {f"layer_{l}": vals for l, vals in r.items()}
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _iter_snapshots(T_z: int) -> List[int]:
    """t = 0, 1, T_z/2, T_z from the proposal's diagnostic schedule."""
    return [0, 1, max(1, T_z // 2), T_z]


def run_probe(
    run_dir: str,
    out_dir: str,
    *,
    seed: int = 0,
    perturb_std: float = 0.1,
    mc_samples_pred: int = 16,
    max_subset: int = 1500,
    include_clean: bool = False,
):
    """Run the matched-pair seed-persistence probe and write outputs."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"[seed-persistence] Loading checkpoint from {run_dir} ...")
    cfg, net = _load_run(run_dir)
    print(f"  hidden_dims={cfg.hidden_dims} activations={cfg.activations}")
    print(f"  output: {cfg.output_likelihood} / {cfg.output_estimator}")
    print(f"  T_z(eval)={cfg.eval_T_z_resolved} S_train={cfg.mc_samples_train}")

    test = load_split(cfg.classes, train=False, seed=cfg.seed)
    print(f"  n_test = {len(test.x)}")

    key = jax.random.PRNGKey(seed)
    key, k_mask = jax.random.split(key)

    print(f"[seed-persistence] MC predictive for subset masks (S={mc_samples_pred}) ...")
    p_mc = _build_p_mc(net, cfg, test.x, k_mask, mc_samples_pred)
    masks_all = _build_subset_masks(p_mc, test.y_idx, conf_clean=0.8, conf_confused=0.4)
    subsets_to_run = ["wrong"]
    if include_clean:
        subsets_to_run.append("clean")
    for name in subsets_to_run:
        n = int(masks_all[name].sum())
        print(f"  subset '{name}': {n} examples ({100 * n / len(test.x):.1f}%)")

    T_z = cfg.eval_T_z_resolved
    snaps = _iter_snapshots(T_z)
    print(f"[seed-persistence] Iteration snapshots: t = {snaps}")
    print(f"[seed-persistence] Perturbation: δ = {perturb_std}·sqrt(v_p^l) per hidden layer")

    L_hidden = net.L_hidden
    interior_layers = list(range(0, L_hidden - 1))  # bottom up to but not including top

    results: Dict[str, Dict[str, object]] = {}
    for name in subsets_to_run:
        mask = masks_all[name]
        n_avail = int(mask.sum())
        if n_avail == 0:
            print(f"[seed-persistence] Subset '{name}' is empty; skipping.")
            continue
        n_use = min(n_avail, max_subset)
        idx_all = np.where(mask)[0]
        rng = np.random.default_rng(seed + _subset_offset(name))
        if n_use < n_avail:
            idx = rng.choice(idx_all, size=n_use, replace=False)
        else:
            idx = idx_all
        x_sub = jnp.asarray(test.x[idx])
        y_idx_sub = jnp.asarray(test.y_idx[idx], dtype=jnp.int32)

        # Use the SAME e_step PRNG for baseline + seeded so the only
        # difference is the initial condition (MC sample noise is matched).
        offset = _subset_offset(name)
        k_estep = jax.random.fold_in(key, 1000 + offset)
        k_perturb = jax.random.fold_in(key, 5000 + offset)

        print(f"[seed-persistence] '{name}' n={n_use} | running baseline E-step ...")
        frozen_b, diag_b = _e_step_traced(net, cfg, x_sub, y_idx_sub, k_estep,
                                          perturb_std=0.0, perturb_key=None)
        print(f"[seed-persistence] '{name}' n={n_use} | running seeded E-step (δ={perturb_std}) ...")
        frozen_s, diag_s = _e_step_traced(net, cfg, x_sub, y_idx_sub, k_estep,
                                          perturb_std=perturb_std, perturb_key=k_perturb)

        # Per-iter, per-layer residuals — both trajectories.
        baseline_trace = _collect_per_iter_residuals(net, x_sub, diag_b, snaps)
        seeded_trace = _collect_per_iter_residuals(net, x_sub, diag_s, snaps)

        # Compute persistence ratios per (layer, metric).
        # ρ^l = (e_abs_seeded - e_abs_baseline)(t=T_z) / (... at t=0)
        persistence: Dict[str, Dict[str, float]] = {}
        for l in range(L_hidden):
            key_l = f"layer_{l}"
            e_delta_0 = (seeded_trace[f"t{snaps[0]}"][key_l]["e_abs"]
                         - baseline_trace[f"t{snaps[0]}"][key_l]["e_abs"])
            e_delta_T = (seeded_trace[f"t{snaps[-1]}"][key_l]["e_abs"]
                         - baseline_trace[f"t{snaps[-1]}"][key_l]["e_abs"])
            if abs(e_delta_0) < 1e-15:
                ratio = float("nan")
            else:
                ratio = e_delta_T / e_delta_0
            persistence[key_l] = {
                "e_delta_t0": float(e_delta_0),
                "e_delta_tT": float(e_delta_T),
                "rho_l": float(ratio),
            }

        # Interior aggregate ρ̄ = min over interior layers.
        if interior_layers:
            interior_rhos = [persistence[f"layer_{l}"]["rho_l"] for l in interior_layers
                             if not math.isnan(persistence[f"layer_{l}"]["rho_l"])]
            rho_bar = min(interior_rhos) if interior_rhos else float("nan")
        else:
            rho_bar = float("nan")

        F_b = {
            "F_initial": float(diag_b.F_initial),
            "F_final": float(diag_b.F_final),
            "F_drop": float(diag_b.F_initial - diag_b.F_final),
        }
        F_s = {
            "F_initial": float(diag_s.F_initial),
            "F_final": float(diag_s.F_final),
            "F_drop": float(diag_s.F_initial - diag_s.F_final),
        }

        results[name] = {
            "n_used": n_use,
            "n_avail": n_avail,
            "baseline_trace": baseline_trace,
            "seeded_trace": seeded_trace,
            "persistence": persistence,
            "rho_bar_interior": rho_bar,
            "F_baseline": F_b,
            "F_seeded": F_s,
        }

        print(f"[seed-persistence] '{name}' persistence ratios per layer:")
        for l in range(L_hidden):
            p = persistence[f"layer_{l}"]
            print(f"  layer {l}: ρ^l = {p['rho_l']:.4f}  "
                  f"(e_delta t=0: {p['e_delta_t0']:.3e} → t={snaps[-1]}: {p['e_delta_tT']:.3e})")
        print(f"  interior min ρ̄ = {rho_bar:.4f}")
        print(f"  baseline F_drop = {F_b['F_drop']:.4e}  seeded F_drop = {F_s['F_drop']:.4e}")

    # --- Write outputs ---
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
        "params": {
            "perturb_std": perturb_std,
            "mc_samples_for_subset_pred": mc_samples_pred,
            "max_subset": max_subset,
            "include_clean": include_clean,
            "seed": seed,
        },
        "iter_snapshots": snaps,
        "interior_layers": interior_layers,
        "results": results,
    }
    json_path = os.path.join(out_dir, "trace.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[seed-persistence] Wrote {json_path}")

    md_path = os.path.join(out_dir, "summary.md")
    _write_summary_md(md_path, payload, L_hidden)
    print(f"[seed-persistence] Wrote {md_path}")


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _write_summary_md(path: str, payload: dict, L_hidden: int):
    lines: List[str] = []
    cfg = payload["config"]
    params = payload["params"]
    snaps = payload["iter_snapshots"]
    interior = payload["interior_layers"]

    lines.append(f"# Seed-persistence trajectory probe — {payload['run_dir']}")
    lines.append("")
    lines.append("Matched-pair diagnostic: same checkpoint, same E-step, same examples;")
    lines.append("the only difference is the initial latent perturbation. Seeded run:")
    lines.append(f"`m_z^l ← m_p^l + {params['perturb_std']}·sqrt(v_p^l)·ξ` at every hidden")
    lines.append("layer. Baseline run: standard predictive init. Both report per-layer")
    lines.append("`mean(|e^l|)` at `t = " + str(snaps) + "` of the E-step.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- hidden_dims: `{cfg['hidden_dims']}`")
    lines.append(f"- activations: `{cfg['activations']}`")
    lines.append(f"- output: `{cfg['output_likelihood']}` / `{cfg['output_estimator']}`")
    lines.append(f"- T_z: {cfg['T_z']}  mc_samples_train: {cfg['mc_samples_train']}  lambda_y: {cfg['lambda_y']}")
    lines.append(f"- perturbation: δ = {params['perturb_std']}·sqrt(v_p^l), per layer (whole-stack seed)")
    lines.append(f"- subset prediction MC samples: {params['mc_samples_for_subset_pred']}")
    lines.append(f"- seed: {params['seed']}")
    lines.append("")

    for subset_name, res in payload["results"].items():
        lines.append(f"## Subset: `{subset_name}`  (n = {res['n_used']})")
        lines.append("")
        # F_DPC trajectory comparison
        F_b = res["F_baseline"]
        F_s = res["F_seeded"]
        lines.append("### F_DPC trajectory")
        lines.append("")
        lines.append("| run | F_initial | F_final | F_drop |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| baseline | {F_b['F_initial']:.4f} | {F_b['F_final']:.4f} | {F_b['F_drop']:.4e} |")
        lines.append(f"| seeded   | {F_s['F_initial']:.4f} | {F_s['F_final']:.4f} | {F_s['F_drop']:.4e} |")
        lines.append("")
        if F_b['F_drop'] != 0:
            f_ratio = F_s['F_drop'] / F_b['F_drop']
            lines.append(f"Seeded ΔF / baseline ΔF = **{f_ratio:.3f}**.")
            if f_ratio > 2.0:
                lines.append("→ Seeded run drops F substantially more than baseline. The E-step is "
                             "actively pushing against the seed (erasure pressure).")
            elif f_ratio < 0.5 or f_ratio < 0:
                lines.append("→ Seeded run drops F substantially less than baseline. The seed may be "
                             "blocking descent or the geometry is unexpected — inspect manually.")
            else:
                lines.append("→ Seeded and baseline F drops are comparable. The seed is approximately "
                             "orthogonal to the descent direction.")
            lines.append("")

        # Per-iter per-layer trajectory
        lines.append("### Per-layer `mean(|e^l|)` trajectories (baseline / seeded / Δ)")
        lines.append("")
        iter_keys = [f"t{t}" for t in snaps]
        for which, label in [("baseline_trace", "baseline"), ("seeded_trace", "seeded")]:
            lines.append(f"**{label}**")
            lines.append("")
            header = "| layer | " + " | ".join(f"t={t}" for t in snaps) + " |"
            divider = "|---|" + "|".join("---:" for _ in snaps) + "|"
            lines.append(header)
            lines.append(divider)
            for l in range(L_hidden):
                row = [f"hidden_{l}"]
                for ik in iter_keys:
                    v = res[which][ik][f"layer_{l}"]["e_abs"]
                    row.append(f"{v:.3e}")
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")
        # Delta table
        lines.append("**seeded − baseline (Δ|e^l|)**")
        lines.append("")
        header = "| layer | " + " | ".join(f"t={t}" for t in snaps) + " |"
        divider = "|---|" + "|".join("---:" for _ in snaps) + "|"
        lines.append(header)
        lines.append(divider)
        for l in range(L_hidden):
            row = [f"hidden_{l}"]
            for ik in iter_keys:
                v_s = res["seeded_trace"][ik][f"layer_{l}"]["e_abs"]
                v_b = res["baseline_trace"][ik][f"layer_{l}"]["e_abs"]
                row.append(f"{v_s - v_b:+.3e}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Persistence ratios table
        lines.append("### Persistence ratios ρ^l = Δ|e^l|(t=T_z) / Δ|e^l|(t=0)")
        lines.append("")
        lines.append("| layer | Δ|e^l|(t=0) | Δ|e^l|(t=T_z) | ρ^l |")
        lines.append("|---|---:|---:|---:|")
        for l in range(L_hidden):
            p = res["persistence"][f"layer_{l}"]
            lines.append(f"| hidden_{l} | {p['e_delta_t0']:.3e} | {p['e_delta_tT']:.3e} | {p['rho_l']:.4f} |")
        lines.append("")
        rho_bar = res["rho_bar_interior"]
        lines.append(f"Interior aggregate (min over l < L_hidden − 1) **ρ̄ = {rho_bar:.4f}**.")
        lines.append("")

        # Verdict
        if math.isnan(rho_bar):
            verdict = ("**N/A** — no interior layers in this architecture (L=1). "
                       "Cannot make the train/no-train call from this run alone.")
        elif rho_bar >= 0.1:
            verdict = ("**Green light** — seed survives within 1 OOM to freeze time. "
                       "The M-step will see a real `r^l ≠ 0` under perturbed-init "
                       "training. Launch the perturbed-init train as specced.")
        elif rho_bar >= 0.01:
            verdict = ("**Yellow** — seed survives but is being attenuated. The "
                       "M-step would see a smaller `r^l` than at injection. Consider "
                       "raising `δ` (e.g. to 0.3·σ) for training, or moving to "
                       "per-E-step re-seed.")
        elif rho_bar > 0:
            verdict = ("**Redirect** — E-step erases the seed before freeze "
                       "(ρ̄ < 0.01). Don't run the one-shot perturbed-init train; "
                       "switch to per-E-step re-seed (standing driver in `step()` "
                       "rather than only in `initial_latents`).")
        elif rho_bar < 0:
            verdict = ("**Anomaly** — Δ|e^l|(t=T_z) and Δ|e^l|(t=0) have opposite "
                       "signs at one or more interior layers. Inspect the per-layer "
                       "table manually; this typically means the seed crossed the "
                       "fixed point or coupled negatively through ReLU gating.")
        else:
            verdict = "**Ambiguous** — manual inspection of the per-layer table recommended."

        if not math.isnan(rho_bar) and rho_bar > 1.0:
            verdict = ("**Instability discovery** — seed *grows* during the E-step "
                       "(ρ̄ > 1). The predictive fixed point is unstable under "
                       "perturbation. Perturbed-init train becomes a confirmation "
                       "of a real instability rather than a hope.")

        lines.append("### Verdict")
        lines.append("")
        lines.append(verdict)
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="experiments.seed_persistence_probe",
        description="Matched-pair seed-persistence trajectory probe for DBPCN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=str, required=True,
                   help="Path to a saved run dir with config.json + weights.npz.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory under reports/ for trace.json + summary.md.")
    p.add_argument("--seed", type=int, default=0,
                   help="PRNG seed for the perturbation and subset subsampling.")
    p.add_argument("--perturb-std", type=float, default=0.1,
                   help="Perturbation scale in units of sqrt(v_p^l).")
    p.add_argument("--mc-samples-pred", type=int, default=16,
                   help="S for the MC predictive used to build subset masks.")
    p.add_argument("--max-subset", type=int, default=1500,
                   help="If the wrong subset is bigger than this, subsample.")
    p.add_argument("--include-clean", action="store_true",
                   help="Also probe the `clean` subset for cross-comparison.")
    args = p.parse_args(argv)
    run_probe(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        perturb_std=args.perturb_std,
        mc_samples_pred=args.mc_samples_pred,
        max_subset=args.max_subset,
        include_clean=args.include_clean,
    )


if __name__ == "__main__":
    main()
