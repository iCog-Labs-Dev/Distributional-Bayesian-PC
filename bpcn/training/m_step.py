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
Its presynaptic feature is psi_1(z^1), with moments propagated through
`psi_moments` (Section 4.5; default psi="identity").
"""
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from ..models.layer import Layer
from ..utils.safe_math import floor_v, clamp_tau
from ..inference.feature_moments import psi_moments
from ..inference.categorical_output import categorical_output_update


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
    r = v_z + e * e - v_p                                  # [B, d_out]

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
    # Categorical-output kwargs (continuation note). Consumed only when
    # `net.output_likelihood == "categorical"`; harmless under "gaussian".
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
    lambda_y: float = 1.0,
):
    """One full M-step for the base BPCN (L_hidden hidden layers + 1 output).

    `data_scale`, `prior_scale` -- see update_layer.

    Output-layer update branches on `net.output_likelihood`:
    - "gaussian"    -> closed-form Eqs. 81-82 via `update_layer(...)` with
                       (y_mean, y_var) as the frozen target.
    - "categorical" -> jax.grad-based update on lambda_y * F_out + gamma_y
                       * KL(q(W_y)||p(W_y)), where F_out is the MEAN or MC
                       softmax NLL (continuation Eqs. 21/27). The hidden
                       layer's `update_layer(...)` call is unchanged in
                       either mode.
    """
    L_hidden = net.L_hidden
    output = net.layers[-1]

    # Per-layer hidden M-step (v2 Section 6.4 / Algorithm 2; continuation
    # Section 6.1 retains this loop unchanged under the categorical extension).
    # For each hidden layer l in 0..L_hidden-1:
    #   - presynaptic moments: (x, 0) if l==0 (v2 Eq. 94), else psi_{l-1}(z^l)
    #     (v2 Eq. 21 / Section 4.5).
    #   - postsynaptic target: frozen.m_zs[l], frozen.v_zs[l] (stop-gradient'd
    #     in the E-step output).
    new_hiddens = []
    hidden_diags = {}
    for l in range(L_hidden):
        if l == 0:
            M_l = jax.lax.stop_gradient(x)
            V_l = jnp.zeros_like(M_l)
        else:
            M_l, V_l = psi_moments(net.activations[l - 1], frozen.m_zs[l - 1], frozen.v_zs[l - 1])
        new_layer, diag = update_layer(
            net.layers[l],
            M=M_l, V=V_l,
            m_z=frozen.m_zs[l], v_z=frozen.v_zs[l],
            alpha=alpha_hidden, gamma=gamma_hidden,
            eta_mu=eta_mu_hidden, eta_tau=eta_tau_hidden,
            data_scale=data_scale, prior_scale=prior_scale,
        )
        new_hiddens.append(new_layer)
        # Per-layer diagnostics key: "hidden_l" for every layer (uniform
        # indexing across all L_hidden values).
        hidden_diags[f"hidden_{l}"] = diag

    # Output-layer update. Presynaptic feature for the output uses psi_L on
    # the *top* hidden latent (continuation Section 4 / Eq. 16).
    if net.output_likelihood == "gaussian":
        # Output layer target = (y_mean, y_var) [Gaussian-logit; assumption I3].
        M_out, V_out = psi_moments(net.activations[-1], frozen.m_zs[-1], frozen.v_zs[-1])
        new_output, out_diags = update_layer(
            output,
            M=M_out, V=V_out,
            m_z=y_mean, v_z=y_var,
            alpha=alpha_output, gamma=gamma_output,
            eta_mu=eta_mu_output, eta_tau=eta_tau_output,
            data_scale=data_scale, prior_scale=prior_scale,
        )
    elif net.output_likelihood == "categorical":
        # Categorical softmax head (continuation Eqs. 43-45 / Section 6.2).
        # `categorical_output_update` consumes only the top latent
        # (frozen.m_z / frozen.v_z via the backward-compat properties).
        if y_idx is None:
            raise ValueError(
                "m_step requires `y_idx` when net.output_likelihood == 'categorical'"
            )
        if key is None:
            key = jax.random.PRNGKey(0)
        new_output, out_diags = categorical_output_update(
            output, frozen, y_idx, key,
            estimator=net.output_estimator,
            S=int(mc_samples_train),
            alpha=alpha_output,
            gamma=gamma_output,
            eta_mu=eta_mu_output,
            eta_tau=eta_tau_output,
            data_scale=data_scale,
            prior_scale=prior_scale,
            lambda_y=lambda_y,
            psi=net.activations[-1],
        )
    else:
        raise ValueError(
            f"unknown output_likelihood: {net.output_likelihood!r}; "
            f"choices: 'gaussian', 'categorical'"
        )

    new_layers = tuple(new_hiddens) + (new_output,)
    new_net = net._replace(layers=new_layers)
    diag_out = dict(hidden_diags)
    diag_out["output"] = out_diags
    return new_net, diag_out
