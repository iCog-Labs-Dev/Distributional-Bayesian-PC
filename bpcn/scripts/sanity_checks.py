"""Sanity checks S1-S8 (plan §12.2) for the base BPCN.

Run:
    python -m bpcn.scripts.sanity_checks

Each check prints PASS/FAIL and a short justification.
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
from ..models.layer import init_layer
from ..inference.e_step import e_step, initial_latents
from ..inference.free_energy import free_energy
from ..inference.shared_energy import shared_free_energy
from ..training.m_step import m_step, update_layer
from ..training.loop import make_batch_step
from ..losses.distributional_kl import gaussian_kl
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
    loop_net_1, _, loop_diag_1, _ = batch_step_1(net, x, y, y_var, jnp.float32(1.0))
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
    _, _, loop_diag_3, _ = batch_step_3(net, x, y, y_var, jnp.float32(1.0))
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


def s12_legacy_reproducibility(cfg):
    """S12: e_step(objective='pc_free_energy') matches a manual legacy descent.

    Verifies that the new dispatching e_step at objective='pc_free_energy'
    produces exactly the same trajectory as the legacy gradient-descent loop
    on free_energy. Tolerance: floating point.
    """
    print("[S12] Legacy backward compatibility (objective='pc_free_energy')")
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

    # Replicate the legacy descent manually.
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
        return _ok(f"legacy match: |dm|={err_m:.3e}, |dv|={err_v:.3e}")
    return _bad(f"legacy diverged: |dm|={err_m:.3e}, |dv|={err_v:.3e} (tol {tol})")


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
        "S12": s12_legacy_reproducibility(cfg),
        "S13": s13_mstep_gradient_identity(cfg),
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
