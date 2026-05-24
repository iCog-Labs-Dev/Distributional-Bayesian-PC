"""Distributional M-step (Algorithm 2): explicit closed-form Eqs. 81-82.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 4.6, Algorithm 2 (Section 6.4).
- Eqs. 60-61: predictive moments.
- Eq. 66: e = m_z - m_p.
- Eq. 68: r = v_z + e^2 - v_p.
- Eq. 81: gradient w.r.t. mu_ij = (N/B) sum_n [ e_ni M_nj / v_p,ni
                                              + r_ni mu_ij V_nj / v_p,ni^2 ]
                                  - gamma * mu_ij / alpha^2.
- Eq. 82: gradient w.r.t. tau_ij = (N/B) sum_n [ r_ni * sigma^2_ij * H^(2)_nj / (2 v_p,ni^2) ]
                                   - (gamma/2)(sigma^2_ij / alpha^2 - 1).

References (extension: shared_energy_dbpcn_extension.pdf):
- Eqs. 47, 54: algebraically identical to Eqs. 81, 82 above (verified by S13).
- Eq. 64: proximal weight damping. Adds rho_w * KL(q(W) || q_old(W)) to the
  M-step objective. Analytic gradient: -rho_w * (mu - mu_old)/sigma_old^2 for mu,
  -rho_w * 0.5 * (sigma^2/sigma_old^2 - 1) for tau (descent direction).
- Eq. 67: bounded variance residual. Replaces r by clip(r/(v_p_safe), -r_max, r_max)
  * v_p_safe in the gradient formulas. Default: no clipping.

Assumption I5 (plan): we implement Eqs. 81-82 explicitly, NOT via autodiff,
so the learning rule is exactly the distributional projection from the
write-up.

SCALING NOTE.  Eq. 81 in the write-up is the gradient of the FULL-DATA loss
estimated from a minibatch: the (N/B) prefactor converts the batch sum into
an unbiased estimator of the full-data sum. We refactor the same gradient
into the standard SGD form, which is equivalent up to a learning-rate
rescaling:

    Eq. 81 form  (full-data scale):   eta_full *  [ (N/B) Sum_b (...) - gamma  * mu/alpha^2 ]
    SGD form    (per-data-point):     eta_sgd  *  [ (1/B) Sum_b (...) - (gamma/N) * mu/alpha^2 ]

with eta_full = eta_sgd / N. We use the SGD form so that eta values live in
the conventional 1e-3 range, but the underlying update is exactly Eq. 81.

The two scales are exposed as `data_scale` (default 1/B) and `prior_scale`
(default 1/N) so a caller can choose either convention. Setting
(data_scale, prior_scale) = (N/B, 1) recovers Eq. 81 verbatim.

The output layer (W_y) uses the SAME update with the target taken as the
Gaussian-logit observation (m_z = y_mean, v_z = y_var)  (assumption I3).
"""
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from ..models.layer import Layer
from ..models.moments import moment_forward
from ..utils.safe_math import floor_v, clamp_tau


class LayerDiagnostics(NamedTuple):
    kl_data: jax.Array        # scalar: post-update mean per-unit KL(q* || q_pred)
    kl_data_before: jax.Array # scalar: pre-update mean per-unit KL for this step
    kl_data_after: jax.Array  # scalar: post-update mean per-unit KL for this step
    kl_data_delta: jax.Array  # scalar: kl_data_after - kl_data_before for this step
    kl_data_loop_before: jax.Array # scalar: pre-update KL before the first inner M-step
    kl_data_loop_after: jax.Array  # scalar: post-update KL after the final inner M-step
    kl_data_loop_delta: jax.Array  # scalar: full inner-loop KL change
    kl_weight: jax.Array      # scalar: mean per-weight KL(q(W) || p(W))
    grad_mu_norm: jax.Array
    grad_tau_norm: jax.Array
    mean_sigma2: jax.Array
    var_residual_pos_frac: jax.Array   # fraction of units with r > 0
    components: dict                   # variance decomposition diagnostics


def update_layer(
    layer: Layer,
    M: jax.Array,            # [B, p_in]   presynaptic mean (post-psi, frozen)
    V: jax.Array,            # [B, p_in]   presynaptic variance
    m_z: jax.Array,          # [B, d_out]  frozen postsynaptic target mean
    v_z: jax.Array,          # [B, d_out]  frozen postsynaptic target variance
    *,
    alpha: float,
    gamma: float,
    eta_mu: float,
    eta_tau: float,
    data_scale: float,
    prior_scale: float,
    r_max=None,
    rho_w: float = 0.0,
    mu_old=None,
    tau_old=None,
) -> Tuple[Layer, LayerDiagnostics]:
    """Apply Eqs. 81-82 to one layer for one minibatch.

    Parameters
    ----------
    data_scale : float
        Multiplier on the batch sum in the data term. Conventional choices:
        - 1/B   : SGD-on-per-data-point loss (default in BaseConfig).
        - N/B   : recovers Eq. 81 letter-for-letter (full-data estimator).
    prior_scale : float
        Multiplier on the KL-to-prior term. Conventional choices:
        - 1/N   : prior weight in per-data-point view (matched to data_scale=1/B).
        - 1     : recovers Eq. 81 letter-for-letter (matched to data_scale=N/B).
    r_max : float or None
        Bounded variance residual (extension Eq. 67). When set, replaces r in
        the gradient formulas by clip(r/(v_p_safe), -r_max, r_max) * v_p_safe.
        Preserves sign; bounds magnitude of the variance update signal.
        Default None = no clipping (legacy behaviour).
    rho_w : float
        Proximal weight damping coefficient (extension Eq. 64). Adds
        -rho_w * (mu - mu_old)/sigma_old^2 to g_mu and -rho_w * 0.5 *
        (sigma^2/sigma_old^2 - 1) to g_tau (the ascent direction, since
        the M-step ascends -KL). Default 0 = no damping.
    mu_old, tau_old : jax.Array or None
        Required when rho_w > 0. Weight posterior parameters at the start of
        the inner M-step loop (extension Section 7.2: "after a single latent
        relaxation"). Default None = no damping reference.
    """
    B = M.shape[0]
    sigma2 = jnp.exp(layer.tau)                            # [d_out, p_in]
    # ----- predictive moments (Eqs. 60-61) -----
    m_p = M @ layer.mu.T                                   # [B, d_out]
    propagated = V @ (layer.mu ** 2).T                     # [B, d_out]
    H2 = M ** 2 + V                                        # [B, p_in]
    epistemic  = H2 @ sigma2.T                             # [B, d_out]
    v_p = layer.beta_inv[None, :] + propagated + epistemic # [B, d_out]
    v_p_safe = floor_v(v_p)

    # ----- distributional error quantities (Eqs. 66, 68) -----
    e = m_z - m_p                                          # [B, d_out]
    r_raw = v_z + e * e - v_p                              # [B, d_out]
    # Bounded variance residual (extension Eq. 67). Preserves sign.
    if r_max is not None:
        rel = r_raw / v_p_safe
        rel_clipped = jnp.clip(rel, -float(r_max), float(r_max))
        r = rel_clipped * v_p_safe
    else:
        r = r_raw

    inv_vp  = 1.0 / v_p_safe
    inv_vp2 = inv_vp * inv_vp

    # ----- data terms of Eqs. 81-82 (batch sum, scaled by data_scale) -----
    g_mu_data_1 = jnp.einsum("bi,bj->ij", e * inv_vp, M)
    g_mu_data_2 = layer.mu * jnp.einsum("bi,bj->ij", r * inv_vp2, V)
    g_mu_data = data_scale * (g_mu_data_1 + g_mu_data_2)

    g_tau_data = data_scale * sigma2 * jnp.einsum("bi,bj->ij", 0.5 * r * inv_vp2, H2)

    # ----- prior KL terms (Eqs. 81 second line, 82 second line) -----
    g_mu_prior  = - prior_scale * gamma * layer.mu / (alpha * alpha)
    g_tau_prior = - prior_scale * 0.5 * gamma * (sigma2 / (alpha * alpha) - 1.0)

    g_mu  = g_mu_data  + g_mu_prior
    g_tau = g_tau_data + g_tau_prior

    # ----- proximal weight damping (extension Eq. 64) -----
    if rho_w > 0.0 and mu_old is not None and tau_old is not None:
        sigma_old2 = jnp.exp(tau_old)
        sigma_old2_safe = floor_v(sigma_old2)
        # Ascent direction (we're maximizing -KL_prox, so subtract grad of KL_prox).
        g_mu_prox = -float(rho_w) * (layer.mu - mu_old) / sigma_old2_safe
        g_tau_prox = -float(rho_w) * 0.5 * (sigma2 / sigma_old2_safe - 1.0)
        g_mu = g_mu + g_mu_prox
        g_tau = g_tau + g_tau_prox

    # ----- gradient ascent on -KL (so we ADD the gradient of the objective) -----
    new_mu  = layer.mu  + eta_mu  * g_mu
    new_tau = clamp_tau(layer.tau + eta_tau * g_tau)

    # ----- diagnostics -----
    v_p_floor = floor_v(v_p)
    kl_data_before = 0.5 * (jnp.log(v_p_floor) - jnp.log(floor_v(v_z))
                            + (v_z + e * e) / v_p_floor - 1.0).mean()
    sigma2_new = jnp.exp(new_tau)
    new_propagated = V @ (new_mu ** 2).T
    new_epistemic = H2 @ sigma2_new.T
    new_v_p = layer.beta_inv[None, :] + new_propagated + new_epistemic
    new_v_p_floor = floor_v(new_v_p)
    new_m_p = M @ new_mu.T
    new_e = m_z - new_m_p
    new_r = v_z + new_e * new_e - new_v_p
    kl_data_after = 0.5 * (jnp.log(new_v_p_floor) - jnp.log(floor_v(v_z))
                           + (v_z + new_e * new_e) / new_v_p_floor - 1.0).mean()
    kl_data_delta = kl_data_after - kl_data_before
    kl_weight = 0.5 * ((sigma2_new + new_mu ** 2) / (alpha * alpha)
                       - 1.0
                       + jnp.log(alpha * alpha) - new_tau).mean()
    diags = LayerDiagnostics(
        kl_data=kl_data_after,
        kl_data_before=kl_data_before,
        kl_data_after=kl_data_after,
        kl_data_delta=kl_data_delta,
        kl_data_loop_before=kl_data_before,
        kl_data_loop_after=kl_data_after,
        kl_data_loop_delta=kl_data_delta,
        kl_weight=kl_weight,
        grad_mu_norm=jnp.sqrt((g_mu ** 2).sum()),
        grad_tau_norm=jnp.sqrt((g_tau ** 2).sum()),
        mean_sigma2=sigma2_new.mean(),
        var_residual_pos_frac=(new_r > 0).mean(),
        components=dict(
            residual_mean=layer.beta_inv.mean(),
            propagated_mean=new_propagated.mean(),
            epistemic_mean=new_epistemic.mean(),
            v_p_mean=new_v_p.mean(),
            v_p_min=new_v_p.min(),
            e_abs_mean=jnp.abs(new_e).mean(),
            r_abs_mean=jnp.abs(new_r).mean(),
            v_p_mean_before=v_p.mean(),
            v_p_min_before=v_p.min(),
            e_abs_mean_before=jnp.abs(e).mean(),
            r_abs_mean_before=jnp.abs(r).mean(),
        ),
    )
    new_layer = Layer(mu=new_mu, tau=new_tau, beta_inv=layer.beta_inv, alpha=layer.alpha)
    return new_layer, diags


def m_step(
    net,                  # Network
    frozen,               # FrozenLatents
    x,                    # [B, d_0]
    y_mean,               # [B, C]
    y_var,                # [B, C]
    *,
    alpha_hidden: float,
    alpha_output: float,
    gamma_hidden: float,
    gamma_output: float,
    eta_mu_hidden: float,
    eta_tau_hidden: float,
    eta_mu_output: float,
    eta_tau_output: float,
    data_scale: float,
    prior_scale: float,
    r_max=None,
    rho_w: float = 0.0,
    net_old=None,
):
    """One full M-step for the base BPCN (1 hidden + 1 output).

    `data_scale`, `prior_scale`, `r_max`, `rho_w` -- see update_layer.
    `net_old` : Network or None
        Reference network for the proximal weight damping term (extension
        Eq. 64). Only used when rho_w > 0. Default None = no damping.
    """
    hidden = net.layers[0]
    output = net.layers[1]

    if net_old is not None:
        mu_old_h, tau_old_h = net_old.layers[0].mu, net_old.layers[0].tau
        mu_old_o, tau_old_o = net_old.layers[1].mu, net_old.layers[1].tau
    else:
        mu_old_h = tau_old_h = mu_old_o = tau_old_o = None

    # Hidden layer target = frozen latent (m_z, v_z); presynaptic = x with V=0  (Eq. 94).
    M_hid = jax.lax.stop_gradient(x)
    V_hid = jnp.zeros_like(M_hid)
    new_hidden, hid_diags = update_layer(
        hidden,
        M=M_hid, V=V_hid,
        m_z=frozen.m_z, v_z=frozen.v_z,
        alpha=alpha_hidden, gamma=gamma_hidden,
        eta_mu=eta_mu_hidden, eta_tau=eta_tau_hidden,
        data_scale=data_scale, prior_scale=prior_scale,
        r_max=r_max, rho_w=rho_w,
        mu_old=mu_old_h, tau_old=tau_old_h,
    )

    # Output layer target = (y_mean, y_var) [Gaussian-logit; assumption I3];
    # presynaptic = frozen latent (m_z, v_z) (identity psi_1).
    new_output, out_diags = update_layer(
        output,
        M=frozen.m_z, V=frozen.v_z,
        m_z=y_mean, v_z=y_var,
        alpha=alpha_output, gamma=gamma_output,
        eta_mu=eta_mu_output, eta_tau=eta_tau_output,
        data_scale=data_scale, prior_scale=prior_scale,
        r_max=r_max, rho_w=rho_w,
        mu_old=mu_old_o, tau_old=tau_old_o,
    )

    new_net = net._replace(layers=(new_hidden, new_output))
    return new_net, {"hidden": hid_diags, "output": out_diags}
