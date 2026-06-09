"""Interior-coupling injection probe (delta-vs-sampled moments).

Background
----------
The hard-subset diagnostic (`experiments/hard_subset_residuals.py`)
confirmed that hidden `|e^l|` is in the 10^-12 to 10^-5 range on every
subset of the L=3 1000-epoch MC checkpoint, including the hardest 20%
of examples. That rules out the "washed-out by averaging" reading of
dormancy. It leaves three hypotheses:

- **World A — dead coupling.** Even if perturbed, the interior latents
  could not transmit a perturbation through the network. The
  vanishingly small `|e^l|` is a structural property of the
  trained-weight Jacobian gain.
- **World B + delta-OK.** The interior coupling is alive; it's only
  dormant because predictive init places the latents at the fixed
  point and nothing pushes them off. The delta-method ReLU moments are
  fine.
- **World B + delta-broken.** The interior coupling is alive in
  principle but the first-order delta-method moment closure
  (`psi_moments`) is silently collapsing the variance-side coupling.
  Exact / sampled ReLU moments would restore non-trivial transmission.

This script injects a known, controlled perturbation at one chosen
hidden layer, holds all other layers and all weights fixed, and
measures whether and how the perturbation propagates — under both the
delta-method and a fresh Monte-Carlo–sampled moment closure. The
three-way verdict drops out of comparing the magnitudes against a
reference injection at the top latent (where we *know* the coupling is
at least weakly alive because the natural E-step reaches |e| ~ 10^-5
there).

References
----------
- v2 Eqs. 60-61: `moment_forward` for (m_p, v_p).
- v2 Eq. 65: per-unit Gaussian KL.
- v2 Section 4.5: psi feature maps and the four moment-propagation
  methods (delta, sigma-point, MC, analytic).
- continuation Eq. 26: `apply_psi_sample` for the exact-psi sample path.

Usage
-----
    python -m experiments.injection_coupling_probe \\
        --run-dir runs/cat_mc_10c2_rerun2 \\
        --out-dir reports/l3_injection_probe \\
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
from bpcn.inference.e_step import initial_latents
from bpcn.inference.feature_moments import (
    apply_psi_sample,
    psi_moments,
    sampled_psi_moments,
)
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.models.moments import moment_forward
from experiments.ood_eval import _load_run


# ---------------------------------------------------------------------------
# Forward primitives that handle both moment schemes uniformly
# ---------------------------------------------------------------------------

def _psi_forward(scheme: str, psi_name: str, m_z, v_z, key, S: int):
    """Return (m_h, v_h) under the chosen moment scheme.

    Both branches are autodiff- and vmap-clean. The sampled branch uses
    the reparam trick so gradients flow through m_z and v_z.
    """
    if scheme == "delta":
        return psi_moments(psi_name, m_z, v_z)
    if scheme == "sampled":
        return sampled_psi_moments(psi_name, m_z, v_z, key, S)
    raise ValueError(f"unknown moment scheme {scheme!r}")


def _layer_predictive(scheme: str, net, layer_idx: int,
                      pre_m_z, pre_v_z, x, key, S: int):
    """Compute (m_p^l, v_p^l) for the transition into `layer_idx`.

    `pre_m_z, pre_v_z` are the LATENT moments of the source for this
    layer (i.e. the latent at layer_idx - 1). For layer_idx == 0, the
    source is the input (deterministic, v_h = 0).

    Returns the predictive moments after passing through psi (if
    interior) + the layer's linear-Gaussian transition via
    `moment_forward`.
    """
    if layer_idx == 0:
        m_h, v_h = x, jnp.zeros_like(x)
    else:
        m_h, v_h = _psi_forward(scheme, net.activations[layer_idx - 1],
                                pre_m_z, pre_v_z, key, S)
    return moment_forward(net.layers[layer_idx], m_h, v_h)


def _output_predictive(scheme: str, net, top_m_z, top_v_z, key, S: int):
    """Compute (m_p^output, v_p^output) from the top hidden latent.

    Mirrors `bpcn/inference/shared_energy.py::_output_predictive` but
    is moment-scheme-aware: the psi_L feature map can be evaluated via
    either delta-method or sampled moments.
    """
    m_h, v_h = _psi_forward(scheme, net.activations[-1],
                            top_m_z, top_v_z, key, S)
    return moment_forward(net.layers[-1], m_h, v_h)


# ---------------------------------------------------------------------------
# Baseline predictive moments per layer (used for delta scaling)
# ---------------------------------------------------------------------------

def _per_layer_predictive_delta(net, x, m_zs, v_zs):
    """Walk the layer chain under delta moments; return per-layer (m_p, v_p).

    These are the predictive moments at predictive init. They scale the
    injection perturbation as
    `delta = delta_scale * sqrt(v_p^target) * eps`.
    """
    L = net.L_hidden
    m_p_list, v_p_list = [], []
    for l in range(L):
        if l == 0:
            M, V = x, jnp.zeros_like(x)
        else:
            M, V = psi_moments(net.activations[l - 1], m_zs[l - 1], v_zs[l - 1])
        m_p_l, v_p_l = moment_forward(net.layers[l], M, V)
        m_p_list.append(m_p_l)
        v_p_list.append(v_p_l)
    return tuple(m_p_list), tuple(v_p_list)


# ---------------------------------------------------------------------------
# The two probes
# ---------------------------------------------------------------------------

def _upward_probe(net, x, m_zs, v_zs, target_layer: int,
                  m_z_target_perturbed, key, S: int, scheme: str):
    """Δ in the layer above's predictive mean, baseline vs perturbed.

    The layer above is:
      - target_layer + 1 if interior (next hidden's m_p)
      - output if target_layer == L_hidden - 1

    Returns per-example L2 norm of the change, averaged over batch.
    """
    L = net.L_hidden

    def _m_p_above(m_z_target):
        if target_layer < L - 1:
            # Forward through layer (target+1) — its source is m_z^target.
            m_p, _ = _layer_predictive(scheme, net, target_layer + 1,
                                       m_z_target, v_zs[target_layer], x,
                                       key, S)
        else:
            # Top layer → output predictive.
            m_p, _ = _output_predictive(scheme, net,
                                        m_z_target, v_zs[target_layer],
                                        key, S)
        return m_p

    m_p_baseline = _m_p_above(m_zs[target_layer])
    m_p_perturbed = _m_p_above(m_z_target_perturbed)
    diff = m_p_perturbed - m_p_baseline       # [B, d_above]
    per_example_norm = jnp.linalg.norm(diff, axis=-1)
    return {
        "mean_norm": float(per_example_norm.mean()),
        "max_norm": float(per_example_norm.max()),
        "shape_above": tuple(int(d) for d in diff.shape[1:]),
    }


def _downward_probe(net, x, m_zs, v_zs, target_layer: int,
                    m_z_target_perturbed, key, S: int, scheme: str):
    """|∂K^target / ∂m_z^{target-1}| backflow gradient.

    K^target = KL(N(m_z^target_perturbed, v_z^target) || N(m_p^target, v_p^target)).
    m_p^target depends on m_z^{target-1} via psi → moment_forward.

    The gradient is taken w.r.t. m_z^{target-1} at its predictive-init
    value (baseline). At baseline, ∂K^target/∂m_z^{target-1} = 0 by
    construction (K is at minimum w.r.t. m_p, and m_p = m_z by
    predictive init — but after we perturb m_z^target, e^target = δ, so
    the gradient becomes -δ/v_p · ∂m_p^target/∂m_z^{target-1}). We
    want the magnitude of that backflow.

    Returns per-example L2 norm of the gradient, averaged over batch.
    """
    if target_layer == 0:
        return None   # no layer below

    L = net.L_hidden
    m_z_below_baseline = m_zs[target_layer - 1]
    v_z_target = v_zs[target_layer]

    def _K_target(m_z_below):
        # Forward through layer `target_layer` using m_z_below as the
        # source. Compute (m_p^target, v_p^target), then K^target with
        # the perturbed m_z^target as the latent.
        m_p_t, v_p_t = _layer_predictive(scheme, net, target_layer,
                                         m_z_below, v_zs[target_layer - 1],
                                         x, key, S)
        kl_obj = gaussian_kl(m_z=m_z_target_perturbed, v_z=v_z_target,
                             m_p=m_p_t, v_p=v_p_t)
        # Sum over units; average over batch to get a scalar.
        return kl_obj.kl.sum(axis=-1).mean()

    grad_fn = jax.grad(_K_target)
    grad = grad_fn(m_z_below_baseline)   # [B, d_{target-1}]
    per_example_norm = jnp.linalg.norm(grad, axis=-1)
    return {
        "mean_norm": float(per_example_norm.mean()),
        "max_norm": float(per_example_norm.max()),
        "shape_below": tuple(int(d) for d in grad.shape[1:]),
    }


# ---------------------------------------------------------------------------
# Probe driver per (target, scheme)
# ---------------------------------------------------------------------------

def _probe_one_cell(net, x, m_zs, v_zs, v_p_per_layer,
                    target_layer: int, scheme: str, delta_scale: float,
                    S: int, key):
    """Run injection + both probes for a single (target, scheme) cell."""
    L = net.L_hidden

    # Build the perturbation: delta_scale * sqrt(v_p^target) * eps,
    # eps ~ N(0, I) per-example, per-unit, fixed PRNG.
    sigma_target = jnp.sqrt(jnp.maximum(v_p_per_layer[target_layer], 0.0))
    key_pert, key_up, key_down = jax.random.split(key, 3)
    eps = jax.random.normal(key_pert, shape=m_zs[target_layer].shape,
                            dtype=m_zs[target_layer].dtype)
    delta = delta_scale * sigma_target * eps      # [B, d_target]
    m_z_target_perturbed = m_zs[target_layer] + delta

    delta_norm = float(jnp.linalg.norm(delta, axis=-1).mean())

    up = _upward_probe(net, x, m_zs, v_zs, target_layer,
                       m_z_target_perturbed, key_up, S, scheme)
    down = _downward_probe(net, x, m_zs, v_zs, target_layer,
                           m_z_target_perturbed, key_down, S, scheme)

    return {
        "target_layer": target_layer,
        "scheme": scheme,
        "delta_scale": delta_scale,
        "delta_mean_norm": delta_norm,
        "upward": up,
        "downward": down,
    }


# ---------------------------------------------------------------------------
# Subset masks (reuses the hard_subset_residuals heuristic)
# ---------------------------------------------------------------------------

def _build_wrong_mask(net, cfg: BaseConfig, x: np.ndarray, y_idx: np.ndarray,
                      key: jax.Array, mc_samples: int, batch_pred: int = 512):
    """Build the `wrong` mask via batched MC predictives."""
    p_mc_chunks = []
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
        p_mc_chunks.append(np.asarray(p_mc_b))
    p_mc = np.concatenate(p_mc_chunks, axis=0)
    return (p_mc.argmax(axis=1) != y_idx), p_mc


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_probe(
    run_dir: str,
    out_dir: str,
    *,
    seed: int = 0,
    delta_scale: float = 0.1,
    mc_samples: int = 32,
    mc_samples_pred: int = 16,
    max_subset: int = 2000,
):
    """Run the four-cell (target × scheme) injection probe and write outputs."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"[injection-probe] Loading checkpoint from {run_dir} ...")
    cfg, net = _load_run(run_dir)
    print(f"  hidden_dims={cfg.hidden_dims} activations={cfg.activations}")
    print(f"  output_likelihood={cfg.output_likelihood} estimator={cfg.output_estimator}")
    L = net.L_hidden

    print(f"[injection-probe] Loading MNIST test split for classes {cfg.classes} ...")
    test = load_split(cfg.classes, train=False, seed=cfg.seed)
    print(f"  n_test = {len(test.x)}")

    print(f"[injection-probe] Building wrong-subset mask via MC (S={mc_samples_pred}) ...")
    key = jax.random.PRNGKey(seed)
    key, k_mask = jax.random.split(key)
    mask_wrong, _p_mc = _build_wrong_mask(net, cfg, test.x, test.y_idx, k_mask,
                                          mc_samples_pred)
    n_avail = int(mask_wrong.sum())
    n_use = min(n_avail, max_subset)
    print(f"  wrong subset: {n_avail} examples; using {n_use}")
    if n_avail == 0:
        raise RuntimeError("wrong subset is empty; nothing to probe")
    rng = np.random.default_rng(seed)
    idx_all = np.where(mask_wrong)[0]
    if n_use < n_avail:
        idx = rng.choice(idx_all, size=n_use, replace=False)
    else:
        idx = idx_all
    x_sub = jnp.asarray(test.x[idx])

    # --- Predictive init: get the baseline latents (m_zs, v_zs) ---
    print("[injection-probe] Computing predictive-init latents ...")
    m0_tup, u0_tup = initial_latents(net, x_sub, cfg.v_init)
    v0_tup = tuple(jnp.exp(u) for u in u0_tup)

    # --- Per-layer predictive moments under delta (used for delta scaling) ---
    print("[injection-probe] Computing per-layer (m_p^l, v_p^l) under delta moments ...")
    _, v_p_per_layer = _per_layer_predictive_delta(net, x_sub, m0_tup, v0_tup)
    for l in range(L):
        sigma_p_l = float(jnp.sqrt(jnp.maximum(v_p_per_layer[l], 0.0)).mean())
        print(f"  layer {l}: mean sqrt(v_p) = {sigma_p_l:.4e} | mean(v_p) = {float(v_p_per_layer[l].mean()):.4e}")

    # --- Decide injection targets ---
    if L >= 2:
        # Interior = layer 1 if L >= 3, else layer 0 if L == 2 (it's the
        # only interior layer for L=2; for L=1 there is no interior).
        # Reference = top hidden layer.
        if L >= 3:
            interior_target = 1
        else:
            interior_target = 0
        top_target = L - 1
        targets = sorted({interior_target, top_target})
    else:
        # L=1: only the top layer; no downward probe possible from the
        # interior because there is no interior. Skip the interior cell.
        targets = [0]

    print(f"[injection-probe] Injection targets: {targets}")

    # --- Run the four-cell grid (or 2-cell for L=1) ---
    results = []
    for target in targets:
        for scheme in ("delta", "sampled"):
            k_cell = jax.random.fold_in(key, 10000 + 100 * target +
                                        (1 if scheme == "sampled" else 0))
            print(f"[injection-probe] target={target} scheme={scheme} ...")
            cell = _probe_one_cell(net, x_sub, m0_tup, v0_tup, v_p_per_layer,
                                   target_layer=target, scheme=scheme,
                                   delta_scale=delta_scale, S=mc_samples,
                                   key=k_cell)
            cell["upward"]["mean_norm_per_unit"] = (
                cell["upward"]["mean_norm"]
                / float(np.prod(cell["upward"]["shape_above"])) ** 0.5
            )
            if cell["downward"] is not None:
                cell["downward"]["mean_norm_per_unit"] = (
                    cell["downward"]["mean_norm"]
                    / float(np.prod(cell["downward"]["shape_below"])) ** 0.5
                )
            results.append(cell)
            print(f"  upward.mean_norm = {cell['upward']['mean_norm']:.4e}")
            if cell["downward"] is not None:
                print(f"  downward.mean_norm = {cell['downward']['mean_norm']:.4e}")
            else:
                print(f"  downward: skipped (target=0)")

    # --- Render outputs ---
    payload = {
        "run_dir": run_dir,
        "config": {
            "hidden_dims": list(cfg.hidden_dims),
            "activations": list(cfg.activations),
            "output_likelihood": cfg.output_likelihood,
            "output_estimator": cfg.output_estimator,
            "T_z": cfg.eval_T_z_resolved,
            "mc_samples_train": cfg.mc_samples_train,
            "lambda_y": cfg.lambda_y,
        },
        "subset": {
            "kind": "wrong",
            "n_avail": int(n_avail),
            "n_used": int(n_use),
        },
        "params": {
            "delta_scale": delta_scale,
            "mc_samples_for_sampled_moments": mc_samples,
            "mc_samples_for_subset_pred": mc_samples_pred,
            "seed": seed,
        },
        "layer_predictive_sigma": {
            f"layer_{l}": float(jnp.sqrt(jnp.maximum(v_p_per_layer[l], 0.0)).mean())
            for l in range(L)
        },
        "results": results,
    }
    json_path = os.path.join(out_dir, "trace.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[injection-probe] Wrote {json_path}")

    md_path = os.path.join(out_dir, "summary.md")
    _write_summary_md(md_path, payload, L)
    print(f"[injection-probe] Wrote {md_path}")


# ---------------------------------------------------------------------------
# Markdown rendering + three-way verdict
# ---------------------------------------------------------------------------

def _write_summary_md(path: str, payload: dict, L: int):
    lines = []
    cfg = payload["config"]
    lines.append(f"# Injection coupling probe — {payload['run_dir']}")
    lines.append("")
    lines.append("Controlled-perturbation probe of interior latent coupling.")
    lines.append("Injects a delta of `delta_scale * sqrt(v_p^target)` (one")
    lines.append(f"random predictive standard deviation, scale={payload['params']['delta_scale']})")
    lines.append("at a chosen layer's mean and measures (a) the change in the")
    lines.append("layer above's predictive mean (upward probe), and (b) the")
    lines.append("backflow gradient `|∂K^target/∂m_z^{target-1}|` (downward probe).")
    lines.append("")
    lines.append("Run under two moment schemes:")
    lines.append("- **delta** — current `psi_moments` first-order delta-method")
    lines.append(f"- **sampled** — Monte Carlo with S={payload['params']['mc_samples_for_sampled_moments']} samples through `apply_psi_sample`")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- hidden_dims: `{cfg['hidden_dims']}`")
    lines.append(f"- activations: `{cfg['activations']}`")
    lines.append(f"- output: `{cfg['output_likelihood']}` / `{cfg['output_estimator']}`")
    lines.append(f"- T_z: {cfg['T_z']}  mc_samples_train: {cfg['mc_samples_train']}")
    lines.append(f"- subset: `wrong` (n_used = {payload['subset']['n_used']} of {payload['subset']['n_avail']})")
    lines.append("")
    lines.append("## Per-layer predictive scale")
    lines.append("")
    lines.append("| layer | mean sqrt(v_p^l) |")
    lines.append("|---|---:|")
    for l in range(L):
        s = payload["layer_predictive_sigma"][f"layer_{l}"]
        lines.append(f"| {l} | {s:.4e} |")
    lines.append("")
    lines.append("## Probe results")
    lines.append("")
    lines.append("| target | scheme | δ mean‖.‖ | upward Δm_p mean‖.‖ | downward |∂K/∂m_z_below| mean‖.‖ |")
    lines.append("|---|---|---:|---:|---:|")
    for c in payload["results"]:
        up = c["upward"]["mean_norm"]
        down = (c["downward"]["mean_norm"]
                if c["downward"] is not None else float("nan"))
        lines.append(
            f"| layer {c['target_layer']} | {c['scheme']} | {c['delta_mean_norm']:.3e} "
            f"| {up:.3e} | {down:.3e} |"
        )
    lines.append("")

    # --- Decision verdict ---
    # Find the top reference cells (target = top_target) and interior cells
    # (target = interior_target = top_target - 2 in L=3, or = 1 if L=3).
    by_key = {(c["target_layer"], c["scheme"]): c for c in payload["results"]}
    top_target = L - 1
    interior_candidates = [c["target_layer"] for c in payload["results"]
                           if c["target_layer"] != top_target]
    if not interior_candidates:
        lines.append("## Verdict")
        lines.append("")
        lines.append("Only one hidden layer in this run (L=1). Cannot compare interior")
        lines.append("vs reference; no three-way decision possible.")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        return
    interior_target = interior_candidates[0]

    # Compute interior/top ratios under each scheme, for both probes.
    def _ratio(metric: str, scheme: str) -> Tuple[float, float, float]:
        ic = by_key[(interior_target, scheme)]
        tc = by_key[(top_target, scheme)]
        if metric == "upward":
            int_v = ic["upward"]["mean_norm"]
            top_v = tc["upward"]["mean_norm"]
        else:
            ic_d = ic["downward"]
            tc_d = tc["downward"]
            if ic_d is None or tc_d is None:
                return (float("nan"), float("nan"), float("nan"))
            int_v = ic_d["mean_norm"]
            top_v = tc_d["mean_norm"]
        if top_v == 0:
            return (int_v, top_v, float("nan"))
        return (int_v, top_v, int_v / top_v)

    up_delta = _ratio("upward", "delta")
    up_samp = _ratio("upward", "sampled")
    dn_delta = _ratio("downward", "delta")
    dn_samp = _ratio("downward", "sampled")

    lines.append("## Interior vs top-reference ratios")
    lines.append("")
    lines.append("| probe | scheme | interior | top | ratio interior/top |")
    lines.append("|---|---|---:|---:|---:|")
    lines.append(f"| upward | delta | {up_delta[0]:.3e} | {up_delta[1]:.3e} | {up_delta[2]:.4f} |")
    lines.append(f"| upward | sampled | {up_samp[0]:.3e} | {up_samp[1]:.3e} | {up_samp[2]:.4f} |")
    lines.append(f"| downward | delta | {dn_delta[0]:.3e} | {dn_delta[1]:.3e} | {dn_delta[2]:.4f} |")
    lines.append(f"| downward | sampled | {dn_samp[0]:.3e} | {dn_samp[1]:.3e} | {dn_samp[2]:.4f} |")
    lines.append("")
    lines.append("Threshold conventions:")
    lines.append("- *alive at delta*: interior/top ratio ≥ 0.1")
    lines.append("- *dead*: interior/top ratio < 0.01")
    lines.append("")

    # --- Three-way verdict heuristic ---
    delta_alive = (
        (not np.isnan(up_delta[2]) and up_delta[2] >= 0.1)
        and (not np.isnan(dn_delta[2]) and dn_delta[2] >= 0.1)
    )
    delta_dead = (
        (not np.isnan(up_delta[2]) and up_delta[2] < 0.01)
        and (not np.isnan(dn_delta[2]) and dn_delta[2] < 0.01)
    )
    samp_alive = (
        (not np.isnan(up_samp[2]) and up_samp[2] >= 0.1)
        and (not np.isnan(dn_samp[2]) and dn_samp[2] >= 0.1)
    )
    samp_dead = (
        (not np.isnan(up_samp[2]) and up_samp[2] < 0.01)
        and (not np.isnan(dn_samp[2]) and dn_samp[2] < 0.01)
    )

    if delta_alive:
        verdict = "**World B + delta-OK**: interior coupling is alive under the current moment closure. Predictive-init dormancy is the bottleneck. *Next experiment: perturbed-init in the E-step.* Sampling rewrite unnecessary."
    elif delta_dead and samp_alive:
        verdict = "**World B + delta-broken**: interior coupling is alive under sampled moments but collapsed by the delta-method closure. *Next experiment: replace `psi_moments` with sampled moments in the E-step inference and re-evaluate the L=3 run.*"
    elif delta_dead and samp_dead:
        verdict = "**World A — dead coupling**: interior coupling is dead under both moment schemes. Perturbed-init in the E-step would seed a perturbation that couples to nothing. *Next experiments are structural*: deeper output anchoring, auxiliary losses attached to interior layers, skip / dense paths inward, stronger source-role weight."
    else:
        verdict = "**Ambiguous** — the interior/top ratios sit between thresholds. Inspect the per-cell table above manually and consider sweeping `delta_scale` to confirm the regime."

    lines.append("## Three-way verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")

    # --- Upward-vs-downward asymmetry caveat ---
    lines.append("## Caveat: upward vs downward asymmetry")
    lines.append("")
    lines.append("The upward probe (Δm_p_above) is more reliable than the downward")
    lines.append("(`|∂K/∂m_z_below|`) because the backflow gradient is itself a product")
    lines.append("of small factors we suspect (small weight means, ReLU gating,")
    lines.append("precision weighting). If upward transmission is healthy but")
    lines.append("downward is near zero, that is specifically an *asymmetric-coupling*")
    lines.append("finding (forward map works, backward credit assignment doesn't) —")
    lines.append("itself a sharp result pointing at structural-credit-assignment levers")
    lines.append("rather than at moment-closure or initialisation fixes.")
    lines.append("")
    lines.append("Watching for the asymmetric-coupling pattern:")
    asym_delta = (
        not np.isnan(up_delta[2]) and up_delta[2] >= 0.1
        and not np.isnan(dn_delta[2]) and dn_delta[2] < 0.01
    )
    asym_samp = (
        not np.isnan(up_samp[2]) and up_samp[2] >= 0.1
        and not np.isnan(dn_samp[2]) and dn_samp[2] < 0.01
    )
    if asym_delta or asym_samp:
        which = []
        if asym_delta:
            which.append("delta")
        if asym_samp:
            which.append("sampled")
        lines.append(
            f"- **Asymmetric coupling DETECTED** under {', '.join(which)}: forward "
            "transmission is healthy but backflow is dead. This is the "
            "structural-credit-assignment finding — points at output-attachment "
            "/ skip-path / auxiliary-loss fixes, not at moment-closure or "
            "init fixes."
        )
    else:
        lines.append(
            "- No asymmetric-coupling pattern detected at the current thresholds."
        )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="experiments.injection_coupling_probe",
        description="Interior-coupling injection probe for DBPCN — delta vs sampled moments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", type=str, required=True,
                   help="Path to a saved run dir with config.json + weights.npz.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory under reports/ for trace.json + summary.md.")
    p.add_argument("--seed", type=int, default=0,
                   help="PRNG seed for perturbation, subsetting, and MC samples.")
    p.add_argument("--delta-scale", type=float, default=0.1,
                   help="Perturbation magnitude in units of sqrt(v_p^target).")
    p.add_argument("--mc-samples", type=int, default=32,
                   help="S for sampled_psi_moments.")
    p.add_argument("--mc-samples-pred", type=int, default=16,
                   help="S for the MC predictive used to build the wrong subset.")
    p.add_argument("--max-subset", type=int, default=2000,
                   help="If the wrong subset is bigger than this, subsample.")
    args = p.parse_args(argv)
    run_probe(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        seed=args.seed,
        delta_scale=args.delta_scale,
        mc_samples=args.mc_samples,
        mc_samples_pred=args.mc_samples_pred,
        max_subset=args.max_subset,
    )


if __name__ == "__main__":
    main()
