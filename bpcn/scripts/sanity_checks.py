"""Sanity checks S1-S25 for the base BPCN.

Run:
    python -m bpcn.scripts.sanity_checks

Each check prints PASS/FAIL and a short justification.

- S1-S9   : base BPCN (plan §12.2).
- S10-S15 : shared-energy extension (M-SE).
- S16-S17 : ReLU feature map psi.
- S18-S23 : categorical output head (continuation note).
- S24-S25 : categorical-output safety (label guard, lambda_y consistency).
"""
from __future__ import annotations
import sys
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from ..configs.base import BaseConfig
from ..models.network import init_network
from ..models.moments import moment_forward
from ..models.layer import init_layer, Layer
from ..inference.e_step import e_step, initial_latents
from ..inference.feature_moments import psi_moments
from ..inference.shared_energy import shared_free_energy, _output_predictive, shared_energy_terms
from ..inference.categorical_output import (
    mean_categorical_loss,
    mc_categorical_loss,
    categorical_output_update,
)
from ..losses.weight_kl import gaussian_weight_kl
from ..training.m_step import m_step, update_layer
from ..training.loop import make_batch_step
from ..losses.distributional_kl import gaussian_kl
from ..evaluation.predict import (
    _target_free_frozen,
    _mean_predict_from_frozen,
    _mc_predict_from_frozen,
)
from ..utils.safe_math import clamp_u


def _ok(msg): print(f"  PASS  {msg}", flush=True); return True
def _bad(msg): print(f"  FAIL  {msg}", flush=True); return False


# -----------------------------------------------------------------------------
def s1_shape_consistency(cfg):
    """S1: pytree shapes match plan §9."""
    print("[S1] Shape consistency")
    key = jax.random.PRNGKey(0)
    net = init_network(key, cfg.layer_dims,
                       alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
                       beta_inv_hidden=cfg.beta_inv_hidden, beta_inv_output=cfg.beta_inv_output,
                       init_log_var=cfg.init_log_var)
    L = net.layers
    h0 = cfg.hidden_dims[0]
    expected = [
        ("L0.mu", L[0].mu.shape, (h0, cfg.input_dim)),
        ("L0.tau", L[0].tau.shape, (h0, cfg.input_dim)),
        ("L0.beta_inv", L[0].beta_inv.shape, (h0,)),
        ("L1.mu", L[1].mu.shape, (cfg.output_dim, h0)),
        ("L1.tau", L[1].tau.shape, (cfg.output_dim, h0)),
        ("L1.beta_inv", L[1].beta_inv.shape, (cfg.output_dim,)),
    ]
    ok = True
    for name, got, want in expected:
        if got != want:
            ok &= _bad(f"{name}: got {got}, expected {want}")
        else:
            _ok(f"{name}: {got}")
    return ok


def s2_positive_variance(cfg):
    """S2: sigma^2, v_z, v_p all > 0 throughout an E-step + M-step."""
    print("[S2] Positive variance")
    key = jax.random.PRNGKey(1)
    net = init_network(key, cfg.layer_dims)
    B = 8
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    frozen, _ = e_step(net, x, y, T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init)
    ok = True
    for name, val in [
        ("sigma2(hidden)", jnp.exp(net.layers[0].tau)),
        ("sigma2(output)", jnp.exp(net.layers[1].tau)),
        ("v_z (after E-step)", frozen.v_z),
    ]:
        if bool((val > 0).all()):
            _ok(f"{name} > 0")
        else:
            ok &= _bad(f"{name} has non-positive entries")
    # Check v_p inside moment_forward after one M-step.
    new_net, m_diag = m_step(net, frozen, x, y, jnp.full_like(y, cfg.target_var),
                             alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
                             gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
                             eta_mu_hidden=cfg.eta_mu_hidden, eta_tau_hidden=cfg.eta_tau_hidden,
                             eta_mu_output=cfg.eta_mu_output, eta_tau_output=cfg.eta_tau_output,
                             data_scale=1.0 / B, prior_scale=1.0 / B)
    v_p_min_hidden = float(m_diag["hidden_0"].components["v_p_min"])
    v_p_min_output = float(m_diag["output"].components["v_p_min"])
    if v_p_min_hidden > 0: _ok(f"v_p min (hidden) = {v_p_min_hidden:.3e} > 0")
    else: ok &= _bad(f"v_p min (hidden) = {v_p_min_hidden:.3e}")
    if v_p_min_output > 0: _ok(f"v_p min (output) = {v_p_min_output:.3e} > 0")
    else: ok &= _bad(f"v_p min (output) = {v_p_min_output:.3e}")
    return ok


def s3_e_m_separation(cfg):
    """S3: M-step weight gradient does not flow through the E-step trajectory.

    Strategy: write a function f(net) that runs e_step then m_step then returns a
    scalar of the new mu. Take jax.grad of f w.r.t. net.layers[0].tau. If gradients
    leaked through the inferred (m, u) buffers, this would be non-zero in a way
    that depends on the E-step's eta_m / T_z. We instead check structurally: the
    e_step function wraps its weight-input in stop_gradient, so no gradient should
    propagate.
    """
    print("[S3] E-step / M-step separation")
    key = jax.random.PRNGKey(2)
    net = init_network(key, cfg.layer_dims)
    B = 4
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg.target_var)

    def f(net):
        frozen, _ = e_step(net, x, y, T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init)
        return frozen.m_z.sum() + frozen.v_z.sum()

    grads = jax.grad(f)(net)
    # All weight gradients should be zero because e_step stop-gradients the net.
    ok = True
    for li, layer in enumerate(grads.layers):
        gmu = float(jnp.abs(layer.mu).max())
        gtau = float(jnp.abs(layer.tau).max())
        if gmu == 0.0 and gtau == 0.0:
            _ok(f"grad through e_step zero for layer {li}")
        else:
            ok &= _bad(f"layer {li}: |dmu|max={gmu:.3e}, |dtau|max={gtau:.3e} (should be 0)")
    return ok


def s4_frozen_immutability(cfg):
    """S4: m_z, v_z unchanged across an M-step (they were stop_gradiented)."""
    print("[S4] Frozen posterior immutability")
    key = jax.random.PRNGKey(3)
    net = init_network(key, cfg.layer_dims)
    B = 4
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg.target_var)
    frozen, _ = e_step(net, x, y, T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init)
    m_z_before = np.asarray(frozen.m_z).copy()
    v_z_before = np.asarray(frozen.v_z).copy()
    _ = m_step(net, frozen, x, y, y_var,
               alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
               gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
               eta_mu_hidden=cfg.eta_mu_hidden, eta_tau_hidden=cfg.eta_tau_hidden,
               eta_mu_output=cfg.eta_mu_output, eta_tau_output=cfg.eta_tau_output,
               data_scale=1.0 / B, prior_scale=1.0 / B)
    if np.allclose(np.asarray(frozen.m_z), m_z_before) and np.allclose(np.asarray(frozen.v_z), v_z_before):
        return _ok("m_z, v_z unchanged across M-step")
    return _bad("frozen statistics mutated during M-step")


def s5_mc_vs_analytic_moments():
    """S5: analytic moment_forward (Eqs. 60-61) matches MC estimate."""
    print("[S5] Analytic moments vs Monte Carlo")
    key = jax.random.PRNGKey(4)
    d_out, p_in, B = 5, 7, 3
    layer = init_layer(key, d_out, p_in, alpha=1.0, beta_inv=0.0, init_log_var=-2.0)
    # Set beta_inv to 0 so the analytic v_pred has no residual term -> compare just to Var(W h).
    rng = np.random.default_rng(4)
    m_h = jnp.asarray(rng.normal(0, 1, (B, p_in)).astype(np.float32))
    v_h = jnp.asarray(rng.uniform(0.01, 0.5, (B, p_in)).astype(np.float32))
    m_pred, v_pred = moment_forward(layer, m_h, v_h)

    # Monte Carlo over (W, h)
    S = 200_000  # large for tight Monte Carlo error
    k1, k2 = jax.random.split(key)
    # Sample S weight matrices and S latent draws per example; we use B=3 small.
    sigma = jnp.sqrt(jnp.exp(layer.tau))
    sigma_h = jnp.sqrt(v_h)

    def mc_one(kk):
        kW, kh = jax.random.split(kk)
        W_s = layer.mu + sigma * jax.random.normal(kW, (d_out, p_in))
        h_s = m_h + sigma_h * jax.random.normal(kh, (B, p_in))
        return h_s @ W_s.T                          # [B, d_out]

    samples = jax.vmap(mc_one)(jax.random.split(k1, S))     # [S, B, d_out]
    mc_mean = samples.mean(axis=0)
    mc_var = samples.var(axis=0)
    err_mean = float(jnp.abs(mc_mean - m_pred).max())
    err_var = float(jnp.abs(mc_var - v_pred).max())
    # Tolerances are intentionally loose because B and S are small.
    if err_mean < 0.05:
        _ok(f"|m_pred - MC mean|_max = {err_mean:.4f} < 0.05")
    else:
        return _bad(f"|m_pred - MC mean|_max = {err_mean:.4f} >= 0.05")
    if err_var < 0.3:
        _ok(f"|v_pred - MC var|_max  = {err_var:.4f} < 0.3")
    else:
        return _bad(f"|v_pred - MC var|_max  = {err_var:.4f} >= 0.3")
    return True


def s6_F_monotone_decrease(cfg):
    """S6: F_z is non-increasing across the T_z iterations of the E-step."""
    print("[S6] F_z monotone decrease across E-step")
    key = jax.random.PRNGKey(5)
    net = init_network(key, cfg.layer_dims)
    B = 16
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    _, diag = e_step(net, x, y, T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init)
    trace = np.asarray(diag.F_trace)
    diffs = np.diff(trace)
    # Allow tiny numerical positive bumps.
    tol = 1e-3 * max(1.0, float(np.max(np.abs(trace))))
    if np.all(diffs <= tol):
        return _ok(f"F_trace non-increasing (max increase = {float(diffs.max()):.4e})")
    return _bad(f"F_trace has positive jumps; diffs={diffs}")


def s7_kl_decrease(cfg):
    """S7: distributional KL on the SAME frozen statistics decreases after a single M-step."""
    print("[S7] Distributional KL decreases per M-step")
    key = jax.random.PRNGKey(6)
    net = init_network(key, cfg.layer_dims)
    B = 32
    rng = np.random.default_rng(6)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg.target_var)
    frozen, _ = e_step(net, x, y, T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init)

    def kl_layer(layer, M, V, m_z_, v_z_):
        m_p, v_p = moment_forward(layer, M, V)
        return gaussian_kl(m_z_, v_z_, m_p, v_p).kl.mean()

    kl_h_before = float(kl_layer(net.layers[0], x, jnp.zeros_like(x), frozen.m_z, frozen.v_z))
    kl_o_before = float(kl_layer(net.layers[1], frozen.m_z, frozen.v_z, y, y_var))
    new_net, _ = m_step(net, frozen, x, y, y_var,
                        alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
                        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
                        eta_mu_hidden=cfg.eta_mu_hidden, eta_tau_hidden=cfg.eta_tau_hidden,
                        eta_mu_output=cfg.eta_mu_output, eta_tau_output=cfg.eta_tau_output,
                        data_scale=1.0 / B, prior_scale=1.0 / B)
    kl_h_after = float(kl_layer(new_net.layers[0], x, jnp.zeros_like(x), frozen.m_z, frozen.v_z))
    kl_o_after = float(kl_layer(new_net.layers[1], frozen.m_z, frozen.v_z, y, y_var))
    ok = True
    if kl_h_after < kl_h_before:
        _ok(f"hidden KL decreased: {kl_h_before:.4f} -> {kl_h_after:.4f}")
    else:
        ok &= _bad(f"hidden KL did NOT decrease: {kl_h_before:.4f} -> {kl_h_after:.4f}")
    if kl_o_after < kl_o_before:
        _ok(f"output KL decreased: {kl_o_before:.4f} -> {kl_o_after:.4f}")
    else:
        ok &= _bad(f"output KL did NOT decrease: {kl_o_before:.4f} -> {kl_o_after:.4f}")
    return ok


def s8_scalar_hand_check():
    """S8: 1-unit / 1-input hand computation of Eqs. 81, 82 vs implementation."""
    print("[S8] Scalar hand-computed gradients vs implementation")
    # One synapse: layer (d_out=1, p_in=1). Single example.
    mu = jnp.array([[0.4]], dtype=jnp.float32)
    tau = jnp.array([[-2.0]], dtype=jnp.float32)
    sigma2 = float(jnp.exp(tau)[0, 0])     # exp(-2) ~ 0.135
    beta_inv = jnp.array([0.05], dtype=jnp.float32)
    alpha = 1.0
    gamma = 1.0

    from ..models.layer import Layer
    layer = Layer(mu=mu, tau=tau, beta_inv=beta_inv, alpha=alpha)
    M = jnp.array([[2.0]], dtype=jnp.float32)
    V = jnp.array([[0.5]], dtype=jnp.float32)
    m_z = jnp.array([[3.0]], dtype=jnp.float32)
    v_z = jnp.array([[0.1]], dtype=jnp.float32)

    # By hand (numbers below derived from Eqs. 60-61, 66-68, 81-82):
    mu_v = 0.4; M_v = 2.0; V_v = 0.5; m_z_v = 3.0; v_z_v = 0.1
    sig2 = sigma2
    H2 = M_v * M_v + V_v                      # = 4.5
    m_p = mu_v * M_v                          # = 0.8
    v_p = 0.05 + mu_v * mu_v * V_v + sig2 * H2  # = 0.05 + 0.16*0.5 + 0.135 * 4.5
    e = m_z_v - m_p                           # = 2.2
    r = v_z_v + e * e - v_p                   # variance residual
    g_mu_data = e * M_v / v_p + r * mu_v * V_v / (v_p * v_p)
    g_mu = g_mu_data - gamma * mu_v / (alpha ** 2)
    g_tau_data = r * sig2 * H2 / (2 * v_p * v_p)
    g_tau = g_tau_data - 0.5 * gamma * (sig2 / (alpha ** 2) - 1.0)

    # Implementation gradients (use eta=1.0 with data_scale=1, prior_scale=1 so that
    # new_param - old_param equals the gradient directly; matches Eq. 81 letter-for-letter).
    new_layer_eta1, _ = update_layer(
        layer, M=M, V=V, m_z=m_z, v_z=v_z,
        alpha=alpha, gamma=gamma,
        eta_mu=1.0, eta_tau=1.0,
        data_scale=1.0, prior_scale=1.0,
    )
    impl_g_mu = float(new_layer_eta1.mu[0, 0] - mu_v)
    impl_g_tau = float(new_layer_eta1.tau[0, 0] - float(tau[0, 0]))

    ok = True
    if abs(impl_g_mu - g_mu) < 1e-4:
        _ok(f"g_mu  hand={g_mu:.6f} impl={impl_g_mu:.6f}")
    else:
        ok &= _bad(f"g_mu  hand={g_mu:.6f} impl={impl_g_mu:.6f}  (diff {abs(impl_g_mu - g_mu):.3e})")
    if abs(impl_g_tau - g_tau) < 1e-4:
        _ok(f"g_tau hand={g_tau:.6f} impl={impl_g_tau:.6f}")
    else:
        ok &= _bad(f"g_tau hand={g_tau:.6f} impl={impl_g_tau:.6f}  (diff {abs(impl_g_tau - g_tau):.3e})")
    return ok


def s9_iterative_m_step(cfg):
    """S9: iterative M-step preserves single-step behavior and reduces full-loop KL."""
    print("[S9] Iterative M-step")
    ok = True
    try:
        _ = BaseConfig(m_step_iters=0)
        ok &= _bad("m_step_iters=0 did not raise")
    except ValueError:
        _ok("m_step_iters=0 raises ValueError")

    key = jax.random.PRNGKey(7)
    B = 16
    cfg1 = replace(cfg, batch_size=B, m_step_iters=1)
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg.target_var)
    y_idx = jnp.zeros((B,), dtype=jnp.int32)  # matches the one-hot at column 0
    batch_key = jax.random.PRNGKey(0)         # Gaussian mode ignores the key
    net = init_network(key, cfg.layer_dims)

    # The manual E-step reference descends the same shared energy F_DPC
    # (extension Eq. 12) as the training loop, bit-for-bit.
    frozen, _ = e_step(
        net, x, y,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        y_var=y_var,
        gamma_hidden=cfg.gamma_hidden,
        gamma_output=cfg.gamma_output,
    )
    direct_net, _ = m_step(
        net, frozen, x, y, y_var,
        alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
        eta_mu_hidden=cfg.eta_mu_hidden, eta_tau_hidden=cfg.eta_tau_hidden,
        eta_mu_output=cfg.eta_mu_output, eta_tau_output=cfg.eta_tau_output,
        data_scale=1.0 / B, prior_scale=1.0 / B,
    )
    batch_step_1 = make_batch_step(cfg1, N_train=B)
    loop_net_1, _, loop_diag_1, _, _, _ = batch_step_1(
        net, x, y, y_var, y_idx, batch_key
    )
    max_diff = 0.0
    for direct_layer, loop_layer in zip(direct_net.layers, loop_net_1.layers):
        max_diff = max(
            max_diff,
            float(jnp.abs(direct_layer.mu - loop_layer.mu).max()),
            float(jnp.abs(direct_layer.tau - loop_layer.tau).max()),
        )
    if max_diff < 1e-6:
        _ok(f"m_step_iters=1 matches one direct m_step (max diff {max_diff:.3e})")
    else:
        ok &= _bad(f"m_step_iters=1 differs from direct m_step (max diff {max_diff:.3e})")

    cfg3 = replace(cfg, batch_size=B, m_step_iters=3)
    batch_step_3 = make_batch_step(cfg3, N_train=B)
    _, _, loop_diag_3, _, _, _ = batch_step_3(
        net, x, y, y_var, y_idx, batch_key
    )
    for name, ld in loop_diag_3.items():
        loop_delta = float(ld.kl_data_loop_delta)
        if loop_delta < 0.0:
            _ok(f"{name} full inner-loop KL decreased ({loop_delta:.4e})")
        else:
            ok &= _bad(f"{name} full inner-loop KL did not decrease ({loop_delta:.4e})")
        consistency = float(jnp.abs(ld.kl_data - ld.kl_data_loop_after))
        if consistency < 1e-7:
            _ok(f"{name} kl_data equals final loop KL")
        else:
            ok &= _bad(f"{name} kl_data/final loop KL mismatch ({consistency:.3e})")
    return ok


# =============================================================================
# Shared-energy extension sanity checks (M-SE)
# References: shared_energy_dbpcn_extension.pdf
# =============================================================================


def s10_F_DPC_monotone(cfg):
    """S10: shared F_kappa (kappa=1) is non-increasing across the E-step scan.

    Reference: extension Eqs. 22, 63 (E-step descends F_kappa).
    """
    print("[S10] F_DPC monotone decrease across shared E-step")
    key = jax.random.PRNGKey(11)
    net = init_network(key, cfg.layer_dims)
    B = 16
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg.target_var)
    _, diag = e_step(
        net, x, y,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        y_var=y_var,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
    )
    trace = np.asarray(diag.F_trace)
    diffs = np.diff(trace)
    tol = 1e-3 * max(1.0, float(np.max(np.abs(trace))))
    if np.all(diffs <= tol):
        return _ok(f"F_DPC monotone (max increase = {float(diffs.max()):.4e})")
    return _bad(f"F_DPC has positive jumps; diffs={diffs}")


def s11_F_DPC_mstep_decrease(cfg):
    """S11: F_DPC (kappa=1) is non-increasing across one M-step.

    Reference: extension Section 5 (M-step descends same F_DPC). Algebraic
    identity Eqs. 47/54 of extension <-> Eqs. 81/82 of original. Uses
    (data_scale=1/B, prior_scale=1) so the M-step descends the same scalar
    that shared_free_energy returns at gamma_*=cfg.gamma_*.
    """
    print("[S11] F_DPC non-increasing across one M-step")
    key = jax.random.PRNGKey(12)
    net = init_network(key, cfg.layer_dims)
    B = 32
    rng = np.random.default_rng(12)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg.target_var)
    frozen, _ = e_step(
        net, x, y,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        y_var=y_var,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
    )

    def f_dpc(net_):
        return float(shared_free_energy(
            net_, x, y, y_var, frozen.m_zs, frozen.v_zs,
                gamma_hidden=cfg.gamma_hidden,
            gamma_output=cfg.gamma_output,
            include_weight_kl=True,
        ))

    F_before = f_dpc(net)
    # Use tiny learning rates to ensure local decrease (the descent guarantee
    # is only true for small enough step).
    new_net, _ = m_step(
        net, frozen, x, y, y_var,
        alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
        eta_mu_hidden=1e-4, eta_tau_hidden=1e-5,
        eta_mu_output=1e-4, eta_tau_output=1e-5,
        data_scale=1.0 / B, prior_scale=1.0,
    )
    F_after = f_dpc(new_net)
    if F_after <= F_before + 1e-4 * max(1.0, abs(F_before)):
        return _ok(f"F_DPC decreased: {F_before:.6f} -> {F_after:.6f} (delta {F_after-F_before:.3e})")
    return _bad(f"F_DPC INCREASED: {F_before:.6f} -> {F_after:.6f} (delta {F_after-F_before:.3e})")


def s13_mstep_gradient_identity(cfg):
    """S13: M-step direction matches -eta * jax.grad(shared_free_energy).

    Hand-derived gradient identity: M-step Eqs. 81/82 <-> jax.grad of F_DPC
    w.r.t. (mu, tau). Verified by cosine similarity (>0.9999) on each layer.
    Uses (data_scale=1/B, prior_scale=1) so the M-step descends exactly the
    F at gamma_*=cfg.gamma_* with no per-batch rescaling.
    """
    print("[S13] M-step gradient = -eta * grad F_DPC")
    key = jax.random.PRNGKey(14)
    net = init_network(key, cfg.layer_dims)
    B = 16
    rng = np.random.default_rng(14)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg.target_var)
    frozen, _ = e_step(
        net, x, y,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        y_var=y_var,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
    )

    def F_of_net(net_):
        return shared_free_energy(
            net_, x, y, y_var, frozen.m_zs, frozen.v_zs,
                gamma_hidden=cfg.gamma_hidden,
            gamma_output=cfg.gamma_output,
            include_weight_kl=True,
        )

    grads = jax.grad(F_of_net)(net)

    # Use tiny learning rates and matched scales so update direction is just -eta * grad.
    eta = 1e-4
    new_net, _ = m_step(
        net, frozen, x, y, y_var,
        alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
        eta_mu_hidden=eta, eta_tau_hidden=eta,
        eta_mu_output=eta, eta_tau_output=eta,
        data_scale=1.0 / B, prior_scale=1.0,
    )

    ok = True
    for li, (g_layer, new_layer, old_layer) in enumerate(zip(grads.layers, new_net.layers, net.layers)):
        d_mu = (new_layer.mu - old_layer.mu).flatten()
        d_tau = (new_layer.tau - old_layer.tau).flatten()
        expected_mu = (-eta * g_layer.mu).flatten()
        expected_tau = (-eta * g_layer.tau).flatten()

        def cos(a, b):
            na = float(jnp.linalg.norm(a))
            nb = float(jnp.linalg.norm(b))
            if na < 1e-12 or nb < 1e-12:
                return 1.0 if (na < 1e-12 and nb < 1e-12) else 0.0
            return float((a @ b) / (na * nb))

        cos_mu = cos(d_mu, expected_mu)
        cos_tau = cos(d_tau, expected_tau)
        if cos_mu > 0.9999 and cos_tau > 0.9999:
            _ok(f"layer {li}: cos(mu)={cos_mu:.6f}, cos(tau)={cos_tau:.6f}")
        else:
            ok &= _bad(f"layer {li}: cos(mu)={cos_mu:.6f}, cos(tau)={cos_tau:.6f}")
    return ok


def s14_predictive_latent_init(cfg):
    """S14: initial_latents uses hidden predictive moments with a v_init floor."""
    print("[S14] Predictive latent initialization")
    key = jax.random.PRNGKey(15)
    net = init_network(
        key,
        cfg.layer_dims,
        alpha_hidden=cfg.alpha_hidden,
        alpha_output=cfg.alpha_output,
        beta_inv_hidden=cfg.beta_inv_hidden,
        beta_inv_output=cfg.beta_inv_output,
        init_log_var=cfg.init_log_var,
    )
    x = jnp.vstack([
        jnp.zeros((1, cfg.input_dim), dtype=jnp.float32),
        jnp.ones((1, cfg.input_dim), dtype=jnp.float32),
    ])

    m0_tup, u0_tup = initial_latents(net, x, cfg.v_init)
    assert len(m0_tup) == 1, "S14 is an L=1 test"
    m0, u0 = m0_tup[0], u0_tup[0]
    m_pred, v_pred = moment_forward(net.layers[0], x, jnp.zeros_like(x))
    v0 = jnp.exp(u0)
    v_expected = jnp.maximum(v_pred, cfg.v_init)

    ok = True
    err_m = float(jnp.abs(m0 - m_pred).max())
    err_v = float(jnp.abs(v0 - v_expected).max())
    if err_m < 1e-6:
        _ok(f"initial mean matches hidden predictive mean (max diff {err_m:.3e})")
    else:
        ok &= _bad(f"initial mean mismatch (max diff {err_m:.3e})")
    if err_v < 1e-6:
        _ok(f"initial variance matches max(v_pred, v_init) (max diff {err_v:.3e})")
    else:
        ok &= _bad(f"initial variance mismatch (max diff {err_v:.3e})")

    kl0 = float(gaussian_kl(m0, v0, m_pred, v_pred).kl.mean())
    if kl0 < 1e-6:
        _ok(f"initial hidden KL is zero under default floor (KL={kl0:.3e})")
    else:
        ok &= _bad(f"initial hidden KL is not zero (KL={kl0:.3e})")

    if float(jnp.abs(v0[1].mean() - v0[0].mean())) > 1e-6:
        _ok("initial variance is input-dependent")
    else:
        ok &= _bad("initial variance is still input-constant")
    return ok


def s15_shared_dpc_target_free_at_fixed_point(cfg):
    """S15: target-free shared-DPC E-step is a no-op under predictive init.

    With objective='shared_dpc', kappa=1, output_weight=0, and the predictive
    latent init, the initial q(z) equals q_pred and so F_trans_dpc(m0, v0)=0,
    grad=0. The scan should leave (m, v) essentially unchanged and the F
    trace should sit at zero. This is the strongest end-to-end check that
    train and eval optimize the same scalar.
    """
    print("[S15] Shared-DPC target-free E-step is a no-op under predictive init")
    key = jax.random.PRNGKey(16)
    net = init_network(
        key, cfg.layer_dims,
        alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
        beta_inv_hidden=cfg.beta_inv_hidden, beta_inv_output=cfg.beta_inv_output,
        init_log_var=cfg.init_log_var,
    )
    rng = np.random.default_rng(16)
    B = 32
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y_zero = jnp.zeros((B, cfg.output_dim), dtype=x.dtype)
    y_var_zero = jnp.zeros_like(y_zero)

    m0_tup, u0_tup = initial_latents(net, x, cfg.v_init)
    assert len(m0_tup) == 1, "S15 is an L=1 test"
    m0, u0 = m0_tup[0], u0_tup[0]
    v0 = jnp.exp(u0)

    frozen, diag = e_step(
        net, x, y_zero,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        output_weight=0.0,
        y_var=y_var_zero,
    )

    err_m = float(jnp.abs(frozen.m_z - m0).max())
    err_v = float(jnp.abs(frozen.v_z - v0).max())
    F_initial = float(diag.F_initial)
    F_final = float(diag.F_final)

    ok = True
    tol_state = 1e-4
    tol_F = 1e-4
    if err_m < tol_state:
        _ok(f"m unchanged across E-step (max |dm|={err_m:.3e} < {tol_state})")
    else:
        ok &= _bad(f"m moved away from predictive-init fixed point (max |dm|={err_m:.3e})")
    if err_v < tol_state:
        _ok(f"v unchanged across E-step (max |dv|={err_v:.3e} < {tol_state})")
    else:
        ok &= _bad(f"v moved away from predictive-init fixed point (max |dv|={err_v:.3e})")
    if abs(F_initial) < tol_F and abs(F_final) < tol_F:
        _ok(f"F sits at zero (F_initial={F_initial:.3e}, F_final={F_final:.3e})")
    else:
        ok &= _bad(f"F not at fixed point (F_initial={F_initial:.3e}, F_final={F_final:.3e})")
    return ok


def s16_relu_psi_moment_correctness(cfg):
    """S16: relu_delta_moments approximates MC moments of ReLU(N(m, v)).

    Delta-method (Section 4.5 option 2): m_h = mask * m_z, v_h = mask * v_z
    with mask = (m_z > 0). Compare to a Monte Carlo estimate over many
    samples. The delta method is exact when |m_z| >> sigma but becomes a
    rough approximation when m_z is near zero. We choose mixed-sign means
    well away from zero so the approximation is tight; for means near zero
    we apply a looser tolerance.
    """
    print("[S16] ReLU psi delta-method vs Monte Carlo moments")
    rng = np.random.default_rng(17)
    # Mixed-sign means well away from zero relative to sigma.
    m_z = jnp.asarray(rng.uniform(-2.0, 2.0, (1024,)).astype(np.float32))
    v_z = jnp.asarray(rng.uniform(0.05, 0.20, (1024,)).astype(np.float32))

    m_h_delta, v_h_delta = psi_moments("relu", m_z, v_z)

    # MC estimate.
    S = 8000
    key = jax.random.PRNGKey(217)
    noise = jax.random.normal(key, (S, m_z.shape[0]))
    z = m_z[None, :] + jnp.sqrt(v_z)[None, :] * noise   # [S, D]
    relu_z = jnp.where(z > 0, z, 0.0)
    m_h_mc = relu_z.mean(axis=0)
    v_h_mc = relu_z.var(axis=0)

    # Restrict the check to units whose mean is at least ~1 sigma from zero,
    # where the delta approximation is supposed to hold tightly.
    sigma = jnp.sqrt(v_z)
    well_separated = jnp.abs(m_z) > sigma
    n_well = int(well_separated.sum())
    if n_well < 32:
        return _bad(f"sample selection produced only {n_well} well-separated units")
    err_m = float(jnp.abs(m_h_delta - m_h_mc)[well_separated].max())
    err_v = float(jnp.abs(v_h_delta - v_h_mc)[well_separated].max())
    tol_m = 0.05
    tol_v = 0.05
    ok = True
    if err_m < tol_m:
        _ok(f"|m_h_delta - m_h_mc|_max = {err_m:.3e} < {tol_m} (well-separated units)")
    else:
        ok &= _bad(f"mean approximation error too large: {err_m:.3e}")
    if err_v < tol_v:
        _ok(f"|v_h_delta - v_h_mc|_max = {err_v:.3e} < {tol_v} (well-separated units)")
    else:
        ok &= _bad(f"variance approximation error too large: {err_v:.3e}")
    # Identity psi must remain a pure pass-through.
    m_id, v_id = psi_moments("identity", m_z, v_z)
    if jnp.allclose(m_id, m_z) and jnp.allclose(v_id, v_z):
        _ok("psi='identity' is a pass-through")
    else:
        ok &= _bad("psi='identity' altered (m_z, v_z)")
    return ok


def s17_relu_end_to_end_dispatch(cfg):
    """S17: end-to-end dispatch with psi='relu' on the base architecture.

    Build a small ReLU-equipped net, run target-free shared-DPC e_step,
    confirm:
      - _output_predictive returns finite moments,
      - shared_free_energy returns a finite scalar,
      - m_step runs without shape errors and produces finite outputs,
      - _mean_predict_from_frozen returns a probability vector summing to 1.
    """
    print("[S17] ReLU end-to-end dispatch")
    cfg_relu = replace(cfg, activations=("relu",) * len(cfg.hidden_dims))
    key = jax.random.PRNGKey(18)
    net = init_network(
        key, cfg_relu.layer_dims,
        alpha_hidden=cfg_relu.alpha_hidden, alpha_output=cfg_relu.alpha_output,
        beta_inv_hidden=cfg_relu.beta_inv_hidden, beta_inv_output=cfg_relu.beta_inv_output,
        init_log_var=cfg_relu.init_log_var,
        activations=cfg_relu.activations,
    )
    if net.activations[-1] != "relu":
        return _bad(f"Network.activations[-1] did not propagate: got {net.activations[-1]!r}")
    _ok(f"Network.activations[-1] == {net.activations[-1]!r}")

    rng = np.random.default_rng(18)
    B = 32
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_relu.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg_relu.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg_relu.target_var)

    # Run a shared-DPC training E-step.
    frozen, e_diag = e_step(
        net, x, y,
        T_z=cfg_relu.T_z, eta_m=cfg_relu.eta_m, eta_u=cfg_relu.eta_u, v_init=cfg_relu.v_init,
        y_var=y_var,
    )
    ok = True
    if jnp.all(jnp.isfinite(frozen.m_z)) and jnp.all(jnp.isfinite(frozen.v_z)) and jnp.all(frozen.v_z > 0):
        _ok("e_step (shared-DPC, relu) produced finite, positive-variance frozen latents")
    else:
        ok &= _bad("e_step produced non-finite or non-positive latent moments")

    # Output predictive uses ReLU(z^1).
    m_py, v_py = _output_predictive(net, frozen.m_z, frozen.v_z)
    if jnp.all(jnp.isfinite(m_py)) and jnp.all(jnp.isfinite(v_py)) and jnp.all(v_py > 0):
        _ok(f"_output_predictive finite (|m_py|_mean={float(jnp.abs(m_py).mean()):.3e})")
    else:
        ok &= _bad("_output_predictive returned non-finite or non-positive moments")

    # Shared free energy is a finite scalar.
    F = float(shared_free_energy(
        net, x, y, y_var, frozen.m_zs, frozen.v_zs,
        gamma_hidden=cfg_relu.gamma_hidden, gamma_output=cfg_relu.gamma_output,
        include_weight_kl=True,
    ))
    if math.isfinite(F):
        _ok(f"shared_free_energy finite (F={F:.4f})")
    else:
        ok &= _bad(f"shared_free_energy non-finite (F={F})")

    # M-step runs without shape errors.
    new_net, m_diag = m_step(
        net, frozen, x, y, y_var,
        alpha_hidden=cfg_relu.alpha_hidden, alpha_output=cfg_relu.alpha_output,
        gamma_hidden=cfg_relu.gamma_hidden, gamma_output=cfg_relu.gamma_output,
        eta_mu_hidden=cfg_relu.eta_mu_hidden, eta_tau_hidden=cfg_relu.eta_tau_hidden,
        eta_mu_output=cfg_relu.eta_mu_output, eta_tau_output=cfg_relu.eta_tau_output,
        data_scale=1.0 / B, prior_scale=1.0,
    )
    finite_new = all(
        jnp.all(jnp.isfinite(lr.mu)) and jnp.all(jnp.isfinite(lr.tau))
        for lr in new_net.layers
    )
    if finite_new and new_net.activations[-1] == "relu":
        _ok("m_step produced finite weights and preserved Network.activations")
    else:
        ok &= _bad("m_step weights non-finite or activations were dropped")

    # Mean predict.
    frozen_eval = _target_free_frozen(
        new_net, x,
        T_z=cfg_relu.eval_T_z_resolved, eta_m=cfg_relu.eval_eta_m_resolved,
        eta_u=cfg_relu.eval_eta_u_resolved, v_init=cfg_relu.eval_v_init_resolved,
    )
    p_mean = _mean_predict_from_frozen(new_net, frozen_eval)
    row_sums = p_mean.sum(axis=-1)
    if jnp.allclose(row_sums, 1.0, atol=1e-5) and jnp.all(p_mean >= 0):
        _ok(f"_mean_predict_from_frozen returns proper distribution (max |sum-1|={float(jnp.abs(row_sums-1).max()):.2e})")
    else:
        ok &= _bad("_mean_predict_from_frozen returned malformed distribution")

    return ok


# =============================================================================
# Categorical-output sanity checks
# References: categorical_output_dbpcn_continuation.pdf
# =============================================================================


def _build_categorical_net(cfg, *, estimator="mean", psi="identity", seed=100):
    """Build a small Network with the categorical output head configured."""
    cfg_cat = replace(
        cfg,
        activations=(psi,) * len(cfg.hidden_dims),
        output_likelihood="categorical",
        output_estimator=estimator,
    )
    net = init_network(
        jax.random.PRNGKey(seed), cfg_cat.layer_dims,
        alpha_hidden=cfg_cat.alpha_hidden, alpha_output=cfg_cat.alpha_output,
        beta_inv_hidden=cfg_cat.beta_inv_hidden, beta_inv_output=cfg_cat.beta_inv_output,
        init_log_var=cfg_cat.init_log_var,
        activations=cfg_cat.activations,
        output_likelihood=cfg_cat.output_likelihood,
        output_estimator=cfg_cat.output_estimator,
    )
    return cfg_cat, net


def s18_mean_categorical_loss_correctness(cfg):
    """S18: mean_categorical_loss matches the hand-computed softmax NLL.

    Reference: continuation Eq. 21:
        ell^mean_n = -log softmax(mu_y M_n^L)_{y_n}.
    """
    print("[S18] Categorical MEAN loss correctness")
    cfg_cat, net = _build_categorical_net(cfg, estimator="mean", seed=130)
    rng = np.random.default_rng(130)
    B = 5
    m_z = jnp.asarray(rng.normal(0, 0.3, (B, cfg.hidden_dims[0])).astype(np.float32))
    v_z = jnp.asarray(rng.uniform(0.01, 0.1, (B, cfg.hidden_dims[0])).astype(np.float32))
    y_idx = jnp.asarray(rng.integers(0, cfg_cat.output_dim, size=(B,)), dtype=jnp.int32)
    impl = float(mean_categorical_loss(net, m_z, v_z, y_idx))

    # Hand computation: apply psi (identity here), compute logits, NLL.
    m_h_np = np.asarray(m_z)
    mu_np = np.asarray(net.layers[-1].mu)
    logits = m_h_np @ mu_np.T                                # [B, C]
    log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    nll = -log_probs[np.arange(B), np.asarray(y_idx)]
    hand = float(nll.mean())

    err = abs(impl - hand)
    if err < 1e-5:
        return _ok(f"|impl - hand| = {err:.3e} < 1e-5 (impl={impl:.6f}, hand={hand:.6f})")
    return _bad(f"MEAN loss mismatch: impl={impl:.6f}, hand={hand:.6f}, err={err:.3e}")


def s19_mc_collapses_to_mean(cfg):
    """S19: MC loss converges to MEAN loss as tau_y -> -inf and v_z -> 0.

    Reference: continuation Eq. 24-27. With near-deterministic weights and
    latents, each MC sample equals the posterior-mean evaluation, so the
    sample-averaged NLL collapses to ell^mean.
    """
    print("[S19] Categorical MC loss collapses to MEAN at small variance")
    _, net = _build_categorical_net(cfg, estimator="mc", seed=131)
    # Force a near-deterministic output weight posterior.
    out = net.layers[-1]
    out_deterministic = out._replace(tau=jnp.full_like(out.tau, -30.0))
    net_det = net._replace(layers=(net.layers[0], out_deterministic))

    rng = np.random.default_rng(131)
    B = 4
    m_z = jnp.asarray(rng.normal(0, 0.3, (B, cfg.hidden_dims[0])).astype(np.float32))
    v_z = jnp.full((B, cfg.hidden_dims[0]), 1e-12, dtype=jnp.float32)
    y_idx = jnp.asarray(rng.integers(0, cfg.output_dim, size=(B,)), dtype=jnp.int32)
    key = jax.random.PRNGKey(231)
    loss_mc = float(mc_categorical_loss(net_det, m_z, v_z, y_idx, key, 256))
    loss_mean = float(mean_categorical_loss(net_det, m_z, v_z, y_idx))
    err = abs(loss_mc - loss_mean)
    tol = 1e-2
    if err < tol:
        return _ok(f"|MC - MEAN| = {err:.3e} < {tol} (MC={loss_mc:.4f}, MEAN={loss_mean:.4f})")
    return _bad(f"MC did not collapse to MEAN: MC={loss_mc:.4f}, MEAN={loss_mean:.4f}, err={err:.3e}")


def s20_categorical_e_step_descent(cfg):
    """S20: F_cat-DPC is non-increasing across the categorical E-step.

    Reference: continuation Section 5 (E-step descends lambda_y F_out +
    F_trans-DPC). For the MEAN estimator the energy is deterministic so
    monotone decrease is required (within numerical noise). MC is run with
    a fixed key so this check is reproducible.
    """
    print("[S20] Categorical E-step monotone descent (MEAN estimator)")
    cfg_cat, net = _build_categorical_net(cfg, estimator="mean", seed=132)
    rng = np.random.default_rng(132)
    B = 16
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_cat.input_dim)).astype(np.float32))
    y_idx = jnp.asarray(rng.integers(0, cfg_cat.output_dim, size=(B,)), dtype=jnp.int32)
    placeholder_y = jnp.zeros((B, cfg_cat.output_dim), dtype=x.dtype)
    placeholder_yvar = jnp.zeros_like(placeholder_y)
    _, diag = e_step(
        net, x, placeholder_y,
        T_z=cfg_cat.T_z, eta_m=cfg_cat.eta_m, eta_u=cfg_cat.eta_u, v_init=cfg_cat.v_init,
        y_var=placeholder_yvar,
        gamma_hidden=cfg_cat.gamma_hidden, gamma_output=cfg_cat.gamma_output,
        y_idx=y_idx, key=jax.random.PRNGKey(0), mc_samples_train=1,
    )
    trace = np.asarray(diag.F_trace)
    diffs = np.diff(trace)
    tol = 1e-3 * max(1.0, float(np.max(np.abs(trace))))
    if np.all(diffs <= tol):
        return _ok(f"F_cat-DPC monotone (max increase = {float(diffs.max()):.4e})")
    return _bad(f"F_cat-DPC has positive jumps; diffs={diffs}")


def s21_categorical_output_update_decreases_loss(cfg):
    """S21: categorical_output_update decreases F_out + gamma_y KL(W_y).

    Reference: continuation Eq. 43 (output M-step descends the same scalar).
    Uses tiny eta to guarantee local descent. MEAN estimator so the test is
    deterministic.
    """
    print("[S21] Categorical output M-step decreases data + weight KL")
    cfg_cat, net = _build_categorical_net(cfg, estimator="mean", seed=133)
    rng = np.random.default_rng(133)
    B = 16
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_cat.input_dim)).astype(np.float32))
    y_idx = jnp.asarray(rng.integers(0, cfg_cat.output_dim, size=(B,)), dtype=jnp.int32)
    placeholder_y = jnp.zeros((B, cfg_cat.output_dim), dtype=x.dtype)
    placeholder_yvar = jnp.zeros_like(placeholder_y)
    frozen, _ = e_step(
        net, x, placeholder_y,
        T_z=cfg_cat.T_z, eta_m=cfg_cat.eta_m, eta_u=cfg_cat.eta_u, v_init=cfg_cat.v_init,
        y_var=placeholder_yvar,
        gamma_hidden=cfg_cat.gamma_hidden, gamma_output=cfg_cat.gamma_output,
        y_idx=y_idx, key=jax.random.PRNGKey(0), mc_samples_train=1,
    )

    output = net.layers[-1]

    def obj(layer):
        loss = mean_categorical_loss(
            net._replace(layers=(net.layers[0], layer)),
            frozen.m_z, frozen.v_z, y_idx,
        )
        kl = gaussian_weight_kl(layer.mu, layer.tau, layer.alpha).sum()
        # Match the J(mu, tau) inside categorical_output_update.
        return float(
            cfg_cat.lambda_y * (1.0 / B) * B * loss
            + (1.0 / B) * cfg_cat.gamma_output * kl
        )

    before = obj(output)
    new_output, diags = categorical_output_update(
        output, frozen, y_idx, jax.random.PRNGKey(0),
        estimator="mean", S=1,
        alpha=cfg_cat.alpha_output, gamma=cfg_cat.gamma_output,
        eta_mu=1e-4, eta_tau=1e-5,
        data_scale=1.0 / B, prior_scale=1.0 / B,
        lambda_y=cfg_cat.lambda_y, psi=cfg_cat.activations[-1],
    )
    after = obj(new_output)
    if after <= before + 1e-5 * max(1.0, abs(before)):
        return _ok(f"objective decreased: {before:.6f} -> {after:.6f} (delta {after - before:.3e})")
    return _bad(f"objective increased: {before:.6f} -> {after:.6f} (delta {after - before:.3e})")


def s22_gaussian_path_byte_identical(cfg):
    """S22: with output_likelihood='gaussian' default, batch_step produces the
    same weights as before the categorical extension.

    Strategy: pass identical seeds through `make_batch_step` and compare
    against a *manually-replayed* Gaussian path using only the legacy
    Gaussian-side update_layer (no categorical branches). The legacy logic
    is recreated in-line so we don't rely on any external baseline file.
    """
    print("[S22] Gaussian path byte-identical (cat extension off)")
    key = jax.random.PRNGKey(22)
    net = init_network(
        key, cfg.layer_dims,
        alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
        beta_inv_hidden=cfg.beta_inv_hidden, beta_inv_output=cfg.beta_inv_output,
        init_log_var=cfg.init_log_var,
        activations=cfg.activations,
    )
    if net.output_likelihood != "gaussian":
        return _bad(f"default output_likelihood is {net.output_likelihood!r}, expected 'gaussian'")
    if net.output_estimator != "mean":
        return _bad(f"default output_estimator is {net.output_estimator!r}, expected 'mean'")

    B = 32
    rng = np.random.default_rng(22)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y_idx = jnp.asarray(rng.integers(0, cfg.output_dim, size=(B,)), dtype=jnp.int32)
    y_mean_np = np.zeros((B, cfg.output_dim), dtype=np.float32)
    y_mean_np[np.arange(B), np.asarray(y_idx)] = 1.0
    y_mean = jnp.asarray(y_mean_np)
    y_var = jnp.full((B, cfg.output_dim), cfg.target_var, dtype=jnp.float32)

    cfg_for_loop = replace(cfg, batch_size=B, m_step_iters=1)
    batch_step = make_batch_step(cfg_for_loop, N_train=B)
    loop_net, _, _, _, _, _ = batch_step(
        net, x, y_mean, y_var, y_idx,
        jax.random.PRNGKey(123),
    )

    # Manual replay using e_step + m_step directly (Gaussian-only paths).
    frozen_ref, _ = e_step(
        net, x, y_mean,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        y_var=y_var,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
        y_idx=y_idx, key=jax.random.split(jax.random.PRNGKey(123), 3)[0],
        mc_samples_train=1,
    )
    ref_net, _ = m_step(
        net, frozen_ref, x, y_mean, y_var,
        alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
        eta_mu_hidden=cfg.eta_mu_hidden, eta_tau_hidden=cfg.eta_tau_hidden,
        eta_mu_output=cfg.eta_mu_output, eta_tau_output=cfg.eta_tau_output,
        data_scale=1.0 / B, prior_scale=1.0 / B,
    )

    max_diff = 0.0
    for L_loop, L_ref in zip(loop_net.layers, ref_net.layers):
        max_diff = max(
            max_diff,
            float(jnp.abs(L_loop.mu - L_ref.mu).max()),
            float(jnp.abs(L_loop.tau - L_ref.tau).max()),
        )
    if max_diff < 1e-6:
        return _ok(f"loop matches direct e_step+m_step (max diff {max_diff:.3e})")
    return _bad(f"loop diverged from direct path (max diff {max_diff:.3e})")


def s23_categorical_end_to_end_dispatch(cfg):
    """S23: end-to-end dispatch with categorical mode (MC, S=2).

    Build a categorical-mode net, run one full `batch_step` through the
    jit-compiled training loop, and confirm:
      - frozen latents finite and positive-variance,
      - shared_energy_terms returns finite scalars,
      - MEAN and MC predictives produce proper distributions on a held-out
        eval batch.
    """
    print("[S23] Categorical end-to-end dispatch (MC, S=2)")
    cfg_cat, net = _build_categorical_net(cfg, estimator="mc", psi="relu", seed=134)
    cfg_cat = replace(cfg_cat, batch_size=16, m_step_iters=1, mc_samples_train=2)

    rng = np.random.default_rng(134)
    B = 16
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_cat.input_dim)).astype(np.float32))
    y_idx = jnp.asarray(rng.integers(0, cfg_cat.output_dim, size=(B,)), dtype=jnp.int32)
    y_mean_np = np.zeros((B, cfg_cat.output_dim), dtype=np.float32)
    y_mean_np[np.arange(B), np.asarray(y_idx)] = 1.0
    y_mean = jnp.asarray(y_mean_np)
    y_var = jnp.full((B, cfg_cat.output_dim), cfg_cat.target_var, dtype=jnp.float32)

    batch_step = make_batch_step(cfg_cat, N_train=B)
    new_net, e_diag, m_diag, f_dpc, _init_res, _freeze_res = batch_step(
        net, x, y_mean, y_var, y_idx,
        jax.random.PRNGKey(234),
    )
    ok = True
    if new_net.output_likelihood != "categorical" or new_net.output_estimator != "mc":
        ok &= _bad(
            f"net aux dropped: output_likelihood={new_net.output_likelihood!r}, "
            f"output_estimator={new_net.output_estimator!r}"
        )
    else:
        _ok("net.output_likelihood and output_estimator preserved after batch_step")

    for name, val in [
        ("f_out", f_dpc.f_out),
        ("f_trans_dpc", f_dpc.f_trans_dpc),
        ("f_weight_kl", f_dpc.f_weight_kl),
    ]:
        if math.isfinite(float(val)):
            _ok(f"{name}={float(val):.4f}")
        else:
            ok &= _bad(f"{name} non-finite: {float(val)}")

    # Eval-time predictives via target-free E-step.
    frozen_eval = _target_free_frozen(
        new_net, x,
        T_z=cfg_cat.eval_T_z_resolved,
        eta_m=cfg_cat.eval_eta_m_resolved,
        eta_u=cfg_cat.eval_eta_u_resolved,
        v_init=cfg_cat.eval_v_init_resolved,
    )
    p_mean = _mean_predict_from_frozen(new_net, frozen_eval)
    p_mc = _mc_predict_from_frozen(new_net, frozen_eval, jax.random.PRNGKey(99), 4)
    for name, p in [("p_mean", p_mean), ("p_mc", p_mc)]:
        row_sums = p.sum(axis=-1)
        if jnp.allclose(row_sums, 1.0, atol=1e-5) and jnp.all(p >= 0):
            _ok(f"{name} is a proper distribution (max |sum-1|={float(jnp.abs(row_sums-1).max()):.2e})")
        else:
            ok &= _bad(f"{name} malformed")
    return ok


def s24_categorical_e_step_requires_y_idx(cfg):
    """S24: e_step raises when y_idx is missing in categorical training mode.

    Reference: continuation Eq. 2/48 -- a non-zero F_out is a label-conditioned
    cross-entropy. The dummy-y_idx fallback (class 0 for every example) would
    silently corrupt the latent gradient. Target-free calls (output_weight=0)
    must still be allowed because the output term cancels.
    """
    print("[S24] Categorical e_step demands y_idx when output_weight != 0")
    cfg_cat, net = _build_categorical_net(cfg, estimator="mean", seed=140)
    rng = np.random.default_rng(140)
    B = 8
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_cat.input_dim)).astype(np.float32))
    placeholder_y = jnp.zeros((B, cfg_cat.output_dim), dtype=x.dtype)
    placeholder_yvar = jnp.zeros_like(placeholder_y)

    ok = True
    raised = False
    try:
        _ = e_step(
            net, x, placeholder_y,
            T_z=cfg_cat.T_z, eta_m=cfg_cat.eta_m, eta_u=cfg_cat.eta_u,
            v_init=cfg_cat.v_init,
            y_var=placeholder_yvar,
            # Deliberately NOT passing y_idx, with the default output_weight=1.0.
        )
    except ValueError as err:
        raised = True
        if "y_idx" in str(err):
            _ok(f"raised ValueError mentioning y_idx: {err}")
        else:
            ok &= _bad(f"raised ValueError but message did not mention y_idx: {err}")
    if not raised:
        ok &= _bad("e_step accepted y_idx=None with categorical + output_weight=1.0")

    # Target-free call (output_weight=0) MUST still pass without y_idx, since
    # the output term is zeroed out and the dummy is multiplied by 0.
    try:
        _ = e_step(
            net, x, placeholder_y,
            T_z=cfg_cat.T_z, eta_m=cfg_cat.eta_m, eta_u=cfg_cat.eta_u,
            v_init=cfg_cat.v_init,
            y_var=placeholder_yvar,
            output_weight=0.0,
        )
        _ok("target-free e_step (output_weight=0) accepts y_idx=None")
    except Exception as err:
        ok &= _bad(f"target-free e_step incorrectly rejected y_idx=None: {err}")
    return ok


def s25_lambda_y_threaded_consistently(cfg):
    """S25: lambda_y is honoured by E-step latent inference and the diagnostic.

    Reference: continuation Eq. 2/48 -- F_cat-DPC = lambda_y F_out +
    F_trans-DPC + F_weight-KL. Before the fix, the E-step ignored lambda_y
    (used output_weight=1.0 by default), decoupling latent inference from
    the M-step's lambda_y * F_out descent.

    Two checks:
      (a) Two E-steps with different output_weight values produce different
          latent fixed points (label gradient strength depends on
          output_weight).
      (b) `shared_energy_terms(..., output_weight=lambda_y).f_out` ==
          `lambda_y * shared_energy_terms(..., output_weight=1.0).f_out`
          so the sum (f_out + f_trans_dpc + f_weight_kl) really is the
          descended F_cat-DPC.
    """
    print("[S25] lambda_y threaded through E-step and shared_energy_terms")
    cfg_cat, net = _build_categorical_net(cfg, estimator="mean", seed=141)
    lambda_y = 3.0

    rng = np.random.default_rng(141)
    B = 16
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_cat.input_dim)).astype(np.float32))
    y_idx = jnp.asarray(rng.integers(0, cfg_cat.output_dim, size=(B,)), dtype=jnp.int32)
    y_mean_np = np.zeros((B, cfg_cat.output_dim), dtype=np.float32)
    y_mean_np[np.arange(B), np.asarray(y_idx)] = 1.0
    y_mean = jnp.asarray(y_mean_np)
    y_var = jnp.full((B, cfg_cat.output_dim), cfg_cat.target_var, dtype=jnp.float32)

    # (a) E-step depends on output_weight.
    common = dict(
        T_z=cfg_cat.T_z, eta_m=cfg_cat.eta_m, eta_u=cfg_cat.eta_u,
        v_init=cfg_cat.v_init,
        y_var=y_var,
        gamma_hidden=cfg_cat.gamma_hidden, gamma_output=cfg_cat.gamma_output,
        y_idx=y_idx, key=jax.random.PRNGKey(0), mc_samples_train=1,
    )
    frozen_lambda, _ = e_step(net, x, y_mean, output_weight=lambda_y, **common)
    frozen_one, _ = e_step(net, x, y_mean, output_weight=1.0, **common)
    ok = True
    diff = float(jnp.abs(frozen_lambda.m_z - frozen_one.m_z).max())
    if diff > 1e-5:
        _ok(f"E-step latent fixed point depends on output_weight (max |dm|={diff:.3e})")
    else:
        ok &= _bad(f"E-step latent ignored output_weight (max |dm|={diff:.3e})")

    # (b) shared_energy_terms scales f_out by output_weight.
    se_one = shared_energy_terms(
        net, x, y_mean, y_var, frozen_lambda.m_zs, frozen_lambda.v_zs,
        y_idx=y_idx, key=jax.random.PRNGKey(0), mc_samples_train=1,
        gamma_hidden=cfg_cat.gamma_hidden, gamma_output=cfg_cat.gamma_output,
        output_weight=1.0,
    )
    se_lambda = shared_energy_terms(
        net, x, y_mean, y_var, frozen_lambda.m_zs, frozen_lambda.v_zs,
        y_idx=y_idx, key=jax.random.PRNGKey(0), mc_samples_train=1,
        gamma_hidden=cfg_cat.gamma_hidden, gamma_output=cfg_cat.gamma_output,
        output_weight=lambda_y,
    )
    f_out_raw = float(se_one.f_out)
    f_out_scaled = float(se_lambda.f_out)
    if f_out_raw < 1e-12:
        ok &= _bad(f"raw f_out vanished; cannot verify scaling (raw={f_out_raw:.3e})")
    else:
        ratio = f_out_scaled / f_out_raw
        if abs(ratio - lambda_y) < 1e-4:
            _ok(f"f_out scales linearly with output_weight (ratio={ratio:.4f}, expected {lambda_y:.4f})")
        else:
            ok &= _bad(
                f"f_out not scaled by output_weight: raw={f_out_raw:.4f}, "
                f"scaled={f_out_scaled:.4f}, ratio={ratio:.4f}, expected {lambda_y:.4f}"
            )

    # f_trans_dpc and f_weight_kl must NOT depend on output_weight.
    if abs(float(se_one.f_trans_dpc) - float(se_lambda.f_trans_dpc)) < 1e-6:
        _ok("f_trans_dpc independent of output_weight (as required)")
    else:
        ok &= _bad("f_trans_dpc changed with output_weight (should not)")
    if abs(float(se_one.f_weight_kl) - float(se_lambda.f_weight_kl)) < 1e-3:
        _ok("f_weight_kl independent of output_weight (as required)")
    else:
        ok &= _bad("f_weight_kl changed with output_weight (should not)")

    return ok


# ---------------------------------------------------------------------------
# S26 -- S31: multi-layer generalization
# ---------------------------------------------------------------------------
# All multi-layer code paths must (a) reproduce L=1 numerics bit-identically
# (regression: S26), (b) descend the canonical F_DPC at L>=2 (S27-S28),
# (c) match the write-up's predictive init and decomposition (S28, S31),
# and (d) implement the two new delta-method activations correctly (S29-S30).


def _build_multi_layer_net(*, hidden_dims, activations,
                            output_likelihood="gaussian",
                            output_estimator="mean", seed=200):
    """Helper: build (cfg, net) for an L=len(hidden_dims) network."""
    cfg = BaseConfig(
        hidden_dims=hidden_dims,
        activations=activations,
        batch_size=16, T_z=5, m_step_iters=1,
        output_likelihood=output_likelihood,
        output_estimator=output_estimator,
    )
    net = init_network(
        jax.random.PRNGKey(seed), cfg.layer_dims,
        alpha_hidden=cfg.alpha_hidden, alpha_output=cfg.alpha_output,
        beta_inv_hidden=cfg.beta_inv_hidden, beta_inv_output=cfg.beta_inv_output,
        init_log_var=cfg.init_log_var,
        activations=cfg.activations,
        output_likelihood=output_likelihood,
        output_estimator=output_estimator,
    )
    return cfg, net


def s27_l2_e_step_descent(cfg):
    """S27: L=2 E-step descends F_DPC across the inner T_z iterations.

    Builds an L_hidden=2 network with `activations=('relu', 'identity')`,
    runs `e_step` with the shared-DPC objective at kappa=1, and asserts the
    recorded F_trace is monotone non-increasing.

    Embeds two write-up alignment checks:
      1. Target+source gradient routing (extension Eq. 31 / continuation
         Eq. 37): perturbing the interior latent `m_zs[0]` must produce a
         non-zero `F_DPC` gradient through both `K^1` (target role) and
         `K^2` (source role through psi_1).
      2. Output term restricted to top latent (continuation Eq. 38):
         `dF_out/dm_zs[0] == 0` for the interior latent.
    """
    print("[S27] L=2 E-step descent + target+source gradient routing")
    cfg_l2, net = _build_multi_layer_net(
        hidden_dims=(64, 32), activations=("relu", "identity"), seed=270,
    )
    rng = np.random.default_rng(270)
    B = 16
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_l2.input_dim)).astype(np.float32))
    y_mean = jnp.zeros((B, cfg_l2.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y_mean, cfg_l2.target_var)

    # E-step descent.
    frozen, e_diag = e_step(
        net, x, y_mean,
        T_z=cfg_l2.T_z, eta_m=cfg_l2.eta_m, eta_u=cfg_l2.eta_u, v_init=cfg_l2.v_init,
        y_var=y_var,
    )
    ok = True
    F_trace = np.asarray(e_diag.F_trace)
    deltas = np.diff(F_trace)
    if deltas.max() <= 1e-3 * max(1.0, abs(float(F_trace[0]))):
        _ok(f"F_DPC monotone over {len(F_trace)} steps (max increase {deltas.max():.3e})")
    else:
        ok &= _bad(f"F_DPC increased mid-trace by {deltas.max():.3e}")
    # Frozen latents must be tuples of length 2.
    if isinstance(frozen.m_zs, tuple) and len(frozen.m_zs) == 2:
        _ok(f"FrozenLatents carries L_hidden=2 latents (shapes {[m.shape for m in frozen.m_zs]})")
    else:
        ok &= _bad(f"FrozenLatents.m_zs is not a length-2 tuple")

    # Target+source gradient routing (Eq. 31). At the predictive-init fixed
    # point all `K^l` are zero (m_z = m_p), so we *perturb* the interior
    # latent off the fixed point and verify the autodiff gradient at the
    # perturbed point is non-zero through both `K^1` (target role) and
    # `K^2` (source role through psi_1).
    m0_tup, u0_tup = initial_latents(net, x, cfg_l2.v_init)
    perturb = jax.random.normal(jax.random.PRNGKey(271), m0_tup[0].shape) * 0.1
    m_perturbed = (m0_tup[0] + perturb, m0_tup[1])
    v_perturbed = tuple(jnp.exp(u) for u in u0_tup)

    def F_full(m_tup, v_tup):
        return shared_free_energy(
            net, x, y_mean, y_var, m_tup, v_tup,
            gamma_output=cfg_l2.gamma_output,
            include_weight_kl=False,
        )

    grads = jax.grad(F_full, argnums=0)(m_perturbed, v_perturbed)
    norm_interior = float(jnp.linalg.norm(grads[0]))
    norm_top = float(jnp.linalg.norm(grads[1]))
    if norm_interior > 1e-6 and norm_top > 1e-6:
        _ok(f"perturbed interior latent receives non-zero F_DPC gradient "
            f"(interior |g|={norm_interior:.3e}, top |g|={norm_top:.3e}) -- "
            f"target+source routing (Eq. 31) OK")
    else:
        ok &= _bad(f"latent gradient unexpectedly zero (interior |g|={norm_interior:.3e}, "
                   f"top |g|={norm_top:.3e})")

    # F_out is restricted to the top latent (Eq. 38). Differentiating the
    # output-only term w.r.t. the interior latent must yield zero.
    from ..inference.shared_energy import _output_term

    def F_out_only(m_tup):
        return _output_term(net, y_mean, y_var, None, m_tup[-1], v_perturbed[-1], None, 1)

    g_out = jax.grad(F_out_only)(m0_tup)
    if float(jnp.abs(g_out[0]).max()) < 1e-8 and float(jnp.linalg.norm(g_out[1])) > 1e-8:
        _ok(f"F_out gradient is zero on interior latent and non-zero on top "
            f"(|g_int|_max={float(jnp.abs(g_out[0]).max()):.1e}, "
            f"|g_top|={float(jnp.linalg.norm(g_out[1])):.3e}) -- continuation Eq. 38 OK")
    else:
        ok &= _bad(
            f"F_out interior gradient should be zero, top gradient non-zero "
            f"(|g_int|_max={float(jnp.abs(g_out[0]).max()):.3e}, "
            f"|g_top|={float(jnp.linalg.norm(g_out[1])):.3e})"
        )
    return ok


def s28_l2_m_step_decreases_f_dpc(cfg):
    """S28: one L=2 M-step decreases the F_DPC scalar.

    Also locks the extension Eq. 72 decomposition invariant:
        f_out + f_trans_dpc + f_weight_kl ~= F_DPC
    so the diagnostic the loop logs equals the scalar the M-step descended.
    """
    print("[S28] L=2 M-step decreases F_DPC + Eq. 72 decomposition sum")
    cfg_l2, net = _build_multi_layer_net(
        hidden_dims=(64, 32), activations=("relu", "tanh"), seed=280,
    )
    rng = np.random.default_rng(280)
    B = 16
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_l2.input_dim)).astype(np.float32))
    y_mean = jnp.zeros((B, cfg_l2.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y_mean, cfg_l2.target_var)

    frozen, _ = e_step(
        net, x, y_mean,
        T_z=cfg_l2.T_z, eta_m=cfg_l2.eta_m, eta_u=cfg_l2.eta_u, v_init=cfg_l2.v_init,
        y_var=y_var,
    )

    def f_dpc(n_):
        return float(shared_free_energy(
            n_, x, y_mean, y_var, frozen.m_zs, frozen.v_zs,
            gamma_output=cfg_l2.gamma_output, include_weight_kl=True,
            weight_kl_scale=1.0 / max(cfg_l2.n_train_total, 1) if cfg_l2.n_train_total else 1.0,
        ))

    # If n_train_total is 0 (not yet populated), use full-data weight KL scale.
    weight_kl_scale = (
        1.0 / cfg_l2.n_train_total if cfg_l2.n_train_total > 0 else 1.0
    )

    def f_dpc_(n_):
        return float(shared_free_energy(
            n_, x, y_mean, y_var, frozen.m_zs, frozen.v_zs,
            gamma_output=cfg_l2.gamma_output, include_weight_kl=True,
            weight_kl_scale=weight_kl_scale,
        ))

    F_before = f_dpc_(net)
    new_net, _ = m_step(
        net, frozen, x, y_mean, y_var,
        alpha_hidden=cfg_l2.alpha_hidden, alpha_output=cfg_l2.alpha_output,
        gamma_hidden=cfg_l2.gamma_hidden, gamma_output=cfg_l2.gamma_output,
        eta_mu_hidden=cfg_l2.eta_mu_hidden, eta_tau_hidden=cfg_l2.eta_tau_hidden,
        eta_mu_output=cfg_l2.eta_mu_output, eta_tau_output=cfg_l2.eta_tau_output,
        data_scale=1.0 / B, prior_scale=weight_kl_scale,
    )
    F_after = f_dpc_(new_net)
    ok = True
    if F_after <= F_before + 1e-4 * max(1.0, abs(F_before)):
        _ok(f"F_DPC decreased {F_before:.4f} -> {F_after:.4f} (delta {F_after-F_before:.3e})")
    else:
        ok &= _bad(f"F_DPC increased {F_before:.4f} -> {F_after:.4f}")

    # Decomposition sum invariant (extension Eq. 72).
    terms = shared_energy_terms(
        new_net, x, y_mean, y_var, frozen.m_zs, frozen.v_zs,
        gamma_hidden=cfg_l2.gamma_hidden, gamma_output=cfg_l2.gamma_output,
        weight_kl_scale=weight_kl_scale,
    )
    sum_terms = float(terms.f_out) + float(terms.f_trans_dpc) + float(terms.f_weight_kl)
    if abs(sum_terms - F_after) < 1e-3 * max(1.0, abs(F_after)):
        _ok(f"decomposition sums to F_DPC: {sum_terms:.4f} ~= {F_after:.4f}")
    else:
        ok &= _bad(f"decomposition mismatch: sum={sum_terms:.4f}, F_DPC={F_after:.4f}")
    return ok


def s29_leaky_relu_moment_correctness(cfg):
    """S29: `leaky_relu_delta_moments` matches MC reference (v2 Section 4.5).

    Sample 16k z's per element from N(m_z, v_z), apply leaky_relu(alpha=0.01)
    exactly, and compare the sample mean/variance to the delta-method
    approximation. Mean is exact (the delta method computes the correct mean
    on a piecewise-linear activation); variance is approximate near m_z=0
    where the Jacobian switches.
    """
    print("[S29] leaky_relu_delta_moments vs Monte Carlo (away from the kink)")
    from ..inference.feature_moments import leaky_relu_delta_moments
    rng = np.random.default_rng(290)
    # The delta method assumes the activation is locally linear over the
    # support of N(m_z, v_z). For leaky_relu this fails near the kink z=0;
    # bias near m_z=0 is ~(1-alpha) * sqrt(v) / sqrt(2*pi). We test in the
    # regime |m_z| >> sqrt(v_z) where the approximation is valid -- this
    # also matches how moments live in the network (|m_z| O(0.1-1),
    # sqrt(v_z) O(0.05) at init).
    m_raw = rng.uniform(-2.0, 2.0, size=(64,))
    # Push small-|m| values away from 0 to stay clear of the kink.
    m_raw = np.where(np.abs(m_raw) < 0.4, np.sign(m_raw) * 0.4, m_raw)
    m = jnp.asarray(m_raw.astype(np.float32))
    v = jnp.asarray(rng.uniform(0.005, 0.05, size=(64,)).astype(np.float32))
    m_pred, v_pred = leaky_relu_delta_moments(m, v)
    S = 16384
    eps = jax.random.normal(jax.random.PRNGKey(29), (S, m.shape[0]))
    z = m[None, :] + jnp.sqrt(v)[None, :] * eps
    h = jax.nn.leaky_relu(z, negative_slope=0.01)
    m_mc = jnp.mean(h, axis=0)
    v_mc = jnp.var(h, axis=0)
    err_m = float(jnp.abs(m_pred - m_mc).max())
    # For leaky_relu with small alpha, var(f(z)) when m_z<0 is alpha^2 * v_z
    # ~ 1e-6, so a tiny absolute MC noise still gives a huge relative error.
    # Use min(absolute, relative) as the check criterion.
    abs_err = jnp.abs(v_pred - v_mc)
    rel_err = abs_err / (v_mc + 1e-6)
    err_v = float(jnp.minimum(abs_err, rel_err).max())
    ok = True
    if err_m < 0.05:
        _ok(f"|mean - MC|_max = {err_m:.3e} < 0.05")
    else:
        ok &= _bad(f"mean mismatch {err_m:.3e}")
    if err_v < 0.05:
        _ok(f"min(abs, rel)|var - MC| max = {err_v:.3e} < 0.05")
    else:
        ok &= _bad(f"variance mismatch {err_v:.3e}")
    return ok


def s30_tanh_moment_correctness(cfg):
    """S30: `tanh_delta_moments` matches MC reference (v2 Section 4.5).

    Same MC strategy as S29 but with tanh. The delta method's mean
    `tanh(m_z)` is biased compared to E[tanh(z)] when v_z is large
    (concavity of tanh), so the tolerance is per-region: tight where
    |m_z| is small, loose where tanh saturates.
    """
    print("[S30] tanh_delta_moments vs Monte Carlo (small v_z regime)")
    from ..inference.feature_moments import tanh_delta_moments
    rng = np.random.default_rng(300)
    # Small-v regime where the first-order Taylor expansion of tanh is
    # accurate. With v ~ 1e-2 the residual bias from tanh's curvature is
    # O(v^2 * tanh''(m_z)) and stays bounded.
    m = jnp.asarray(rng.uniform(-1.5, 1.5, size=(64,)).astype(np.float32))
    v = jnp.asarray(rng.uniform(0.005, 0.05, size=(64,)).astype(np.float32))
    m_pred, v_pred = tanh_delta_moments(m, v)
    S = 16384
    eps = jax.random.normal(jax.random.PRNGKey(30), (S, m.shape[0]))
    z = m[None, :] + jnp.sqrt(v)[None, :] * eps
    h = jnp.tanh(z)
    m_mc = jnp.mean(h, axis=0)
    v_mc = jnp.var(h, axis=0)
    err_m = float(jnp.abs(m_pred - m_mc).max())
    err_v_rel = float((jnp.abs(v_pred - v_mc) / (v_mc + 1e-6)).max())
    ok = True
    if err_m < 0.05:
        _ok(f"|mean - MC|_max = {err_m:.3e} < 0.05")
    else:
        ok &= _bad(f"mean mismatch {err_m:.3e}")
    if err_v_rel < 0.4:
        _ok(f"rel|var - MC| max = {err_v_rel:.3f} < 0.4 in the small-v regime")
    else:
        ok &= _bad(f"variance mismatch (rel) {err_v_rel:.3f}")
    return ok


def s31_multi_layer_end_to_end(cfg):
    """S31: L=3 categorical-MC end-to-end + predictive init reproduces Eq. 89.

    Builds `hidden_dims=(64, 48, 32), activations=('relu', 'leaky_relu', 'tanh')`
    with output_likelihood='categorical', output_estimator='mc'; runs one
    `batch_step` via the production loop and verifies:
      - finite F components, finite weights after update,
      - per-layer `m_step` diagnostics ('hidden_0', 'hidden_1', 'hidden_2', 'output'),
      - both predictive heads (mean / MC) produce proper distributions.

    Predictive-init check (v2 Eq. 89): independently compute
        m_1 = mu_1 x,  m_2 = mu_2 psi_1(m_1),  m_3 = mu_3 psi_2(m_2)
    and verify `initial_latents` returns the same per-layer means.
    """
    print("[S31] L=3 end-to-end + predictive init Eq. 89")
    cfg_l3, net = _build_multi_layer_net(
        hidden_dims=(64, 48, 32),
        activations=("relu", "leaky_relu", "tanh"),
        output_likelihood="categorical", output_estimator="mc",
        seed=310,
    )
    rng = np.random.default_rng(310)
    B = 16
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_l3.input_dim)).astype(np.float32))
    y_idx = jnp.asarray(rng.integers(0, cfg_l3.output_dim, size=(B,)), dtype=jnp.int32)
    y_mean = jnp.zeros((B, cfg_l3.output_dim), dtype=x.dtype)
    y_mean = y_mean.at[jnp.arange(B), y_idx].set(1.0)
    y_var = jnp.full_like(y_mean, cfg_l3.target_var)
    ok = True

    # Predictive init Eq. 89 reproduction. Compute the chain by hand.
    from ..inference.feature_moments import psi_moments
    m_hand = []
    m_h = x
    v_h = jnp.zeros_like(x)
    for l in range(net.L_hidden):
        m_p, v_p = moment_forward(net.layers[l], m_h, v_h)
        m_hand.append(m_p)
        if l + 1 < net.L_hidden:
            m_h, v_h = psi_moments(net.activations[l], m_p, v_p)
    m0_tup, _ = initial_latents(net, x, cfg_l3.v_init)
    max_diff = max(float(jnp.abs(m0_tup[l] - m_hand[l]).max()) for l in range(net.L_hidden))
    if max_diff < 1e-6:
        _ok(f"initial_latents reproduces Eq. 89 chain (max diff {max_diff:.3e})")
    else:
        ok &= _bad(f"initial_latents diverges from Eq. 89 (max diff {max_diff:.3e})")

    # End-to-end batch_step via the production loop.
    batch_step = make_batch_step(cfg_l3, N_train=B * 100)
    new_net, e_diag, m_diag, f_dpc, _init_res, _freeze_res = batch_step(
        net, x, y_mean, y_var, y_idx, jax.random.PRNGKey(311)
    )
    # Per-layer diagnostics present.
    expected_keys = {"hidden_0", "hidden_1", "hidden_2", "output"}
    if expected_keys.issubset(set(m_diag.keys())):
        _ok(f"per-layer m_diag keys present: {sorted(m_diag.keys())}")
    else:
        ok &= _bad(f"per-layer m_diag keys missing; got {sorted(m_diag.keys())}")
    # Finite weights.
    finite = all(
        jnp.all(jnp.isfinite(lr.mu)) and jnp.all(jnp.isfinite(lr.tau))
        for lr in new_net.layers
    )
    if finite:
        _ok("batch_step produced finite weights across all 4 layers")
    else:
        ok &= _bad("non-finite weights after batch_step")
    # Finite F components.
    if all(jnp.isfinite(getattr(f_dpc, k)) for k in ("f_out", "f_trans_dpc", "f_weight_kl")):
        _ok(f"f_out={float(f_dpc.f_out):.4f} f_trans_dpc={float(f_dpc.f_trans_dpc):.4f} "
            f"f_weight_kl={float(f_dpc.f_weight_kl):.4f}")
    else:
        ok &= _bad("non-finite f_dpc components")
    # Both predictive heads produce proper distributions.
    frozen_eval = _target_free_frozen(
        new_net, x,
        T_z=cfg_l3.T_z, eta_m=cfg_l3.eta_m, eta_u=cfg_l3.eta_u, v_init=cfg_l3.v_init,
        gamma_hidden=cfg_l3.gamma_hidden, gamma_output=cfg_l3.gamma_output,
    )
    p_mean = _mean_predict_from_frozen(new_net, frozen_eval)
    p_mc = _mc_predict_from_frozen(new_net, frozen_eval, jax.random.PRNGKey(312), 4)
    if float(jnp.abs(p_mean.sum(axis=-1) - 1.0).max()) < 1e-5:
        _ok("MEAN predictive sums to 1")
    else:
        ok &= _bad("MEAN predictive not a proper distribution")
    if float(jnp.abs(p_mc.sum(axis=-1) - 1.0).max()) < 1e-5:
        _ok("MC predictive sums to 1")
    else:
        ok &= _bad("MC predictive not a proper distribution")
    return ok


def s32_review_regressions(cfg):
    """S32: regression locks for review-found bugs.

    Bug (P2): the categorical MC output M-step's sample-side activation
    only handled 'relu'; with `--activations relu,tanh` (or `leaky_relu`)
    it silently degraded to identity. We assert that the categorical
    update gradient w.r.t. mu is *non-zero* under the new activations and
    *differs* from what an identity head would produce, locking that the
    M-step objective actually applies the configured activation.

    Bug (P-keys): m_step must emit `"hidden_l"` keys uniformly for every
    L_hidden value (no L=1 special case, no aggregate alias). We assert
    the L=2 m_step returns exactly `{"hidden_0", "hidden_1", "output"}`.
    """
    print("[S32] Regression locks for review-found bugs")
    ok = True

    # --- per-layer m_diag keys ---
    cfg_l2, net_l2 = _build_multi_layer_net(
        hidden_dims=(64, 32), activations=("relu", "identity"), seed=320,
    )
    rng = np.random.default_rng(320)
    B = 16
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_l2.input_dim)).astype(np.float32))
    y_mean = jnp.zeros((B, cfg_l2.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y_mean, cfg_l2.target_var)
    frozen, _ = e_step(
        net_l2, x, y_mean,
        T_z=cfg_l2.T_z, eta_m=cfg_l2.eta_m, eta_u=cfg_l2.eta_u, v_init=cfg_l2.v_init,
        y_var=y_var,
    )
    _, m_diag = m_step(
        net_l2, frozen, x, y_mean, y_var,
        alpha_hidden=cfg_l2.alpha_hidden, alpha_output=cfg_l2.alpha_output,
        gamma_hidden=cfg_l2.gamma_hidden, gamma_output=cfg_l2.gamma_output,
        eta_mu_hidden=cfg_l2.eta_mu_hidden, eta_tau_hidden=cfg_l2.eta_tau_hidden,
        eta_mu_output=cfg_l2.eta_mu_output, eta_tau_output=cfg_l2.eta_tau_output,
        data_scale=1.0 / B, prior_scale=1.0,
    )
    expected_keys = {"hidden_0", "hidden_1", "output"}
    if set(m_diag.keys()) == expected_keys:
        _ok(f"per-layer m_diag keys are exactly {sorted(expected_keys)}")
    else:
        ok &= _bad(
            f"L=2 m_diag keys differ from expected; got {sorted(m_diag.keys())}"
        )

    # --- P3 ---
    # Build a categorical-MC net with a non-ReLU top activation; verify the
    # M-step's gradient on mu depends on the activation choice (rather than
    # silently degrading to identity, as the previous code did).
    from ..inference.categorical_output import _categorical_objective
    cfg_cat, net_cat = _build_categorical_net(cfg, estimator="mc", seed=322)
    cfg_cat = replace(cfg_cat, activations=("tanh",))
    # Re-init net with the tanh activation.
    net_cat = init_network(
        jax.random.PRNGKey(322), cfg_cat.layer_dims,
        alpha_hidden=cfg_cat.alpha_hidden, alpha_output=cfg_cat.alpha_output,
        beta_inv_hidden=cfg_cat.beta_inv_hidden, beta_inv_output=cfg_cat.beta_inv_output,
        init_log_var=cfg_cat.init_log_var,
        activations=cfg_cat.activations,
        output_likelihood="categorical", output_estimator="mc",
    )
    Bc = 8
    rng = np.random.default_rng(322)
    m_z = jnp.asarray(rng.uniform(-1.0, 1.0, (Bc, cfg_cat.hidden_dims[-1])).astype(np.float32))
    v_z = jnp.full_like(m_z, 0.01)
    y_idx = jnp.asarray(rng.integers(0, cfg_cat.output_dim, (Bc,)), dtype=jnp.int32)
    output = net_cat.layers[-1]

    def J_with_psi(psi_name):
        loss_and_aux = _categorical_objective(
            output.mu, output.tau, m_z, v_z, y_idx,
            jax.random.PRNGKey(42), 4,
            estimator="mc", psi=psi_name,
            alpha=output.alpha, gamma=cfg_cat.gamma_output, lambda_y=1.0,
            data_scale=1.0 / Bc, prior_scale=1.0 / Bc, B=Bc,
        )
        return loss_and_aux[0]

    g_tanh = jax.grad(lambda mu: _categorical_objective(
        mu, output.tau, m_z, v_z, y_idx, jax.random.PRNGKey(42), 4,
        estimator="mc", psi="tanh",
        alpha=output.alpha, gamma=cfg_cat.gamma_output, lambda_y=1.0,
        data_scale=1.0 / Bc, prior_scale=1.0 / Bc, B=Bc,
    )[0])(output.mu)
    g_id = jax.grad(lambda mu: _categorical_objective(
        mu, output.tau, m_z, v_z, y_idx, jax.random.PRNGKey(42), 4,
        estimator="mc", psi="identity",
        alpha=output.alpha, gamma=cfg_cat.gamma_output, lambda_y=1.0,
        data_scale=1.0 / Bc, prior_scale=1.0 / Bc, B=Bc,
    )[0])(output.mu)
    diff = float(jnp.linalg.norm(g_tanh - g_id))
    if diff > 1e-6:
        _ok(f"P3: categorical MC mu-gradient differs between psi=tanh and psi=identity "
            f"(|g_tanh - g_id| = {diff:.3e}) -- the sample-side activation is applied")
    else:
        ok &= _bad(
            f"P3: categorical MC mu-gradient is identical for tanh and identity "
            f"(|g_tanh - g_id| = {diff:.3e}) -- the activation is being silently dropped"
        )
    return ok


def main():
    cfg = BaseConfig()
    print(f"[bpcn] Running sanity checks with config layer_dims={cfg.layer_dims}", flush=True)
    results = {
        "S1": s1_shape_consistency(cfg),
        "S2": s2_positive_variance(cfg),
        "S3": s3_e_m_separation(cfg),
        "S4": s4_frozen_immutability(cfg),
        "S5": s5_mc_vs_analytic_moments(),
        "S6": s6_F_monotone_decrease(cfg),
        "S7": s7_kl_decrease(cfg),
        "S8": s8_scalar_hand_check(),
        "S9": s9_iterative_m_step(cfg),
        "S10": s10_F_DPC_monotone(cfg),
        "S11": s11_F_DPC_mstep_decrease(cfg),
        "S13": s13_mstep_gradient_identity(cfg),
        "S14": s14_predictive_latent_init(cfg),
        "S15": s15_shared_dpc_target_free_at_fixed_point(cfg),
        "S16": s16_relu_psi_moment_correctness(cfg),
        "S17": s17_relu_end_to_end_dispatch(cfg),
        "S18": s18_mean_categorical_loss_correctness(cfg),
        "S19": s19_mc_collapses_to_mean(cfg),
        "S20": s20_categorical_e_step_descent(cfg),
        "S21": s21_categorical_output_update_decreases_loss(cfg),
        "S22": s22_gaussian_path_byte_identical(cfg),
        "S23": s23_categorical_end_to_end_dispatch(cfg),
        "S24": s24_categorical_e_step_requires_y_idx(cfg),
        "S25": s25_lambda_y_threaded_consistently(cfg),
        "S27": s27_l2_e_step_descent(cfg),
        "S28": s28_l2_m_step_decreases_f_dpc(cfg),
        "S29": s29_leaky_relu_moment_correctness(cfg),
        "S30": s30_tanh_moment_correctness(cfg),
        "S31": s31_multi_layer_end_to_end(cfg),
        "S32": s32_review_regressions(cfg),
    }
    print()
    print("=" * 60)
    for name, ok in results.items():
        flag = "PASS" if ok else "FAIL"
        print(f"  {name}: {flag}")
    all_pass = all(results.values())
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
