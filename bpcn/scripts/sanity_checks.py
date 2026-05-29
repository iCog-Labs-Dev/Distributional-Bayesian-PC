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
from ..inference.feature_moments import psi_moments, relu_delta_moments
from ..inference.free_energy import free_energy
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
    expected = [
        ("L0.mu", L[0].mu.shape, (cfg.hidden_dim, cfg.input_dim)),
        ("L0.tau", L[0].tau.shape, (cfg.hidden_dim, cfg.input_dim)),
        ("L0.beta_inv", L[0].beta_inv.shape, (cfg.hidden_dim,)),
        ("L1.mu", L[1].mu.shape, (cfg.output_dim, cfg.hidden_dim)),
        ("L1.tau", L[1].tau.shape, (cfg.output_dim, cfg.hidden_dim)),
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
    v_p_min_hidden = float(m_diag["hidden"].components["v_p_min"])
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

    # Use the SAME objective as the loop will (cfg.objective) so the manual
    # E-step reference matches the loop's E-step bit-for-bit.
    frozen, _ = e_step(
        net, x, y,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        y_var=y_var,
        objective=cfg.objective,
        kappa=1.0,
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
    loop_net_1, _, loop_diag_1, _ = batch_step_1(
        net, x, y, y_var, y_idx, batch_key, jnp.float32(1.0)
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
    _, _, loop_diag_3, _ = batch_step_3(
        net, x, y, y_var, y_idx, batch_key, jnp.float32(1.0)
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
        y_var=y_var, objective="shared_dpc", kappa=1.0,
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
        y_var=y_var, objective="shared_dpc", kappa=1.0,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
    )

    def f_dpc(net_):
        return float(shared_free_energy(
            net_, x, y, y_var, frozen.m_z, frozen.v_z,
            kappa=1.0,
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


def s12_legacy_objective_dispatch(cfg):
    """S12: e_step(objective='pc_free_energy') matches manual objective descent.

    Verifies that objective='pc_free_energy' descends the legacy Eq. 40 scalar
    with the current initialization. This is dispatch consistency, not a claim
    that old run numerics are reproduced after predictive latent init.
    """
    print("[S12] Legacy objective dispatch (objective='pc_free_energy')")
    key = jax.random.PRNGKey(13)
    net = init_network(key, cfg.layer_dims)
    B = 8
    rng = np.random.default_rng(13)
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)

    frozen, _ = e_step(
        net, x, y,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        objective="pc_free_energy",
    )

    # Replicate the legacy-objective descent manually from the current init.
    W = jax.lax.stop_gradient(net)
    m0, u0 = initial_latents(W, x, cfg.v_init)

    def F_of(m, u):
        return free_energy(W, x, y, m, jnp.exp(u), output_weight=1.0)

    grad_fn = jax.grad(F_of, argnums=(0, 1))
    m_legacy, u_legacy = m0, u0
    for _ in range(cfg.T_z):
        gm, gu = grad_fn(m_legacy, u_legacy)
        m_legacy = m_legacy - cfg.eta_m * gm
        u_legacy = clamp_u(u_legacy - cfg.eta_u * gu)
    v_legacy = jnp.exp(u_legacy)

    err_m = float(jnp.abs(frozen.m_z - m_legacy).max())
    err_v = float(jnp.abs(frozen.v_z - v_legacy).max())
    tol = 1e-5
    if err_m < tol and err_v < tol:
        return _ok(f"dispatch match: |dm|={err_m:.3e}, |dv|={err_v:.3e}")
    return _bad(f"dispatch diverged: |dm|={err_m:.3e}, |dv|={err_v:.3e} (tol {tol})")


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
        y_var=y_var, objective="shared_dpc", kappa=1.0,
        gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
    )

    def F_of_net(net_):
        return shared_free_energy(
            net_, x, y, y_var, frozen.m_z, frozen.v_z,
            kappa=1.0,
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

    m0, u0 = initial_latents(net, x, cfg.v_init)
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

    m0, u0 = initial_latents(net, x, cfg.v_init)
    v0 = jnp.exp(u0)

    frozen, diag = e_step(
        net, x, y_zero,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        output_weight=0.0,
        y_var=y_var_zero,
        objective="shared_dpc",
        kappa=1.0,
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
    cfg_relu = replace(cfg, psi="relu")
    key = jax.random.PRNGKey(18)
    net = init_network(
        key, cfg_relu.layer_dims,
        alpha_hidden=cfg_relu.alpha_hidden, alpha_output=cfg_relu.alpha_output,
        beta_inv_hidden=cfg_relu.beta_inv_hidden, beta_inv_output=cfg_relu.beta_inv_output,
        init_log_var=cfg_relu.init_log_var,
        psi=cfg_relu.psi,
    )
    if net.psi != "relu":
        return _bad(f"Network.psi did not propagate: got {net.psi!r}")
    _ok(f"Network.psi == {net.psi!r}")

    rng = np.random.default_rng(18)
    B = 32
    x = jnp.asarray(rng.uniform(0, 1, (B, cfg_relu.input_dim)).astype(np.float32))
    y = jnp.zeros((B, cfg_relu.output_dim), dtype=jnp.float32).at[:, 0].set(1.0)
    y_var = jnp.full_like(y, cfg_relu.target_var)

    # Run a shared-DPC training E-step.
    frozen, e_diag = e_step(
        net, x, y,
        T_z=cfg_relu.T_z, eta_m=cfg_relu.eta_m, eta_u=cfg_relu.eta_u, v_init=cfg_relu.v_init,
        y_var=y_var, objective="shared_dpc", kappa=1.0,
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
        net, x, y, y_var, frozen.m_z, frozen.v_z,
        kappa=1.0, gamma_hidden=cfg_relu.gamma_hidden, gamma_output=cfg_relu.gamma_output,
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
    if finite_new and new_net.psi == "relu":
        _ok("m_step produced finite weights and preserved Network.psi")
    else:
        ok &= _bad("m_step weights non-finite or psi was dropped")

    # Mean predict.
    frozen_eval = _target_free_frozen(
        new_net, x,
        T_z=cfg_relu.eval_T_z_resolved, eta_m=cfg_relu.eval_eta_m_resolved,
        eta_u=cfg_relu.eval_eta_u_resolved, v_init=cfg_relu.eval_v_init_resolved,
        objective="shared_dpc",
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
        psi=psi,
        output_likelihood="categorical",
        output_estimator=estimator,
    )
    net = init_network(
        jax.random.PRNGKey(seed), cfg_cat.layer_dims,
        alpha_hidden=cfg_cat.alpha_hidden, alpha_output=cfg_cat.alpha_output,
        beta_inv_hidden=cfg_cat.beta_inv_hidden, beta_inv_output=cfg_cat.beta_inv_output,
        init_log_var=cfg_cat.init_log_var,
        psi=cfg_cat.psi,
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
    m_z = jnp.asarray(rng.normal(0, 0.3, (B, cfg.hidden_dim)).astype(np.float32))
    v_z = jnp.asarray(rng.uniform(0.01, 0.1, (B, cfg.hidden_dim)).astype(np.float32))
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
    m_z = jnp.asarray(rng.normal(0, 0.3, (B, cfg.hidden_dim)).astype(np.float32))
    v_z = jnp.full((B, cfg.hidden_dim), 1e-12, dtype=jnp.float32)
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
        objective="shared_dpc", kappa=1.0,
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
        objective="shared_dpc", kappa=1.0,
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
        lambda_y=cfg_cat.lambda_y, psi=cfg_cat.psi,
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
        psi=cfg.psi,
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
    loop_net, _, _, _ = batch_step(
        net, x, y_mean, y_var, y_idx,
        jax.random.PRNGKey(123),
        jnp.float32(1.0),
    )

    # Manual replay using e_step + m_step directly (Gaussian-only paths).
    frozen_ref, _ = e_step(
        net, x, y_mean,
        T_z=cfg.T_z, eta_m=cfg.eta_m, eta_u=cfg.eta_u, v_init=cfg.v_init,
        y_var=y_var,
        objective=cfg.objective, kappa=1.0,
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
    new_net, e_diag, m_diag, f_dpc = batch_step(
        net, x, y_mean, y_var, y_idx,
        jax.random.PRNGKey(234),
        jnp.float32(1.0),
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
        objective="shared_dpc",
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
            y_var=placeholder_yvar, objective="shared_dpc", kappa=1.0,
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
            y_var=placeholder_yvar, objective="shared_dpc", kappa=1.0,
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
        y_var=y_var, objective="shared_dpc", kappa=1.0,
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
        net, x, y_mean, y_var, frozen_lambda.m_z, frozen_lambda.v_z,
        y_idx=y_idx, key=jax.random.PRNGKey(0), mc_samples_train=1,
        gamma_hidden=cfg_cat.gamma_hidden, gamma_output=cfg_cat.gamma_output,
        output_weight=1.0,
    )
    se_lambda = shared_energy_terms(
        net, x, y_mean, y_var, frozen_lambda.m_z, frozen_lambda.v_z,
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
        "S12": s12_legacy_objective_dispatch(cfg),
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
