"""Distributional M-step """

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from bpcn.models.layer import Layer
from bpcn.utils.safe_math import floor_v, clamp_tau
from bpcn.inference.feature_moments import psi_moments
from bpcn.inference.categorical_output import categorical_output_update


class LayerDiagnostics(NamedTuple):
    kl_data: jax.Array        # post-update mean per-unit KL(q* || q_pred)
    kl_data_before: jax.Array #  pre-update mean per-unit KL for this step
    kl_data_after: jax.Array  # post-update mean per-unit KL for this step
    kl_data_delta: jax.Array  # kl_data_after - kl_data_before for this step
    kl_data_loop_before: jax.Array #  pre-update KL before the first inner M-step
    kl_data_loop_after: jax.Array  #  post-update KL after the final inner M-step
    kl_data_loop_delta: jax.Array  #  full inner-loop KL change
    kl_weight: jax.Array      # mean per-weight KL(q(W) || p(W))
    grad_mu_norm: jax.Array            #  ‖g_μ‖ (total, data + prior)
    grad_tau_norm: jax.Array           #  ‖g_τ‖ (total, data + prior)
    grad_mu_data_norm: jax.Array
    grad_mu_prior_norm: jax.Array
    grad_tau_data_norm: jax.Array
    grad_tau_prior_norm: jax.Array
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
    gamma_mu_override=None,
) -> Tuple[Layer, LayerDiagnostics]:
    """Apply a single M-step update to a layer's parameters.

    This performs a gradient-ascent step on the negative KL objective for a
    single fully-connected layer given frozen presynaptic/postsynaptic
    moments. It computes predictive moments, distributional error
    quantities, data and prior gradients for both the weight means (mu)
    and log-variances (tau), and applies learning-rate-scaled updates.
    """
    B = M.shape[0]
    sigma2 = jnp.exp(layer.tau)                            # [d_out, p_in]
    
    #  predictive moments 
    m_p = M @ layer.mu.T                                   # [B, d_out]
    propagated = V @ (layer.mu ** 2).T                     # [B, d_out]
    H2 = M ** 2 + V                                        # [B, p_in]
    epistemic  = H2 @ sigma2.T                             # [B, d_out]
    v_p = layer.beta_inv[None, :] + propagated + epistemic # [B, d_out]
    v_p_safe = floor_v(v_p)

    # distributional error quantities
    e = m_z - m_p                                          # [B, d_out]
    r = v_z + e * e - v_p                                  # [B, d_out]

    inv_vp  = 1.0 / v_p_safe
    inv_vp2 = inv_vp * inv_vp

    g_mu_data_1 = jnp.einsum("bi,bj->ij", e * inv_vp, M)
    g_mu_data_2 = layer.mu * jnp.einsum("bi,bj->ij", r * inv_vp2, V)
    g_mu_data = data_scale * (g_mu_data_1 + g_mu_data_2)

    g_tau_data = data_scale * sigma2 * jnp.einsum("bi,bj->ij", 0.5 * r * inv_vp2, H2)

    gamma_mu_eff = gamma if gamma_mu_override is None else gamma_mu_override
    g_mu_prior  = - prior_scale * gamma_mu_eff * layer.mu / (alpha * alpha)
    g_tau_prior = - prior_scale * 0.5 * gamma * (sigma2 / (alpha * alpha) - 1.0)

    g_mu  = g_mu_data  + g_mu_prior
    g_tau = g_tau_data + g_tau_prior

    # gradient ascent on -KL (so we ADD the gradient of the objective)
    new_mu  = layer.mu  + eta_mu  * g_mu
    new_tau = clamp_tau(layer.tau + eta_tau * g_tau)

    # diagnostics
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
        grad_mu_data_norm=jnp.sqrt((g_mu_data ** 2).sum()),
        grad_mu_prior_norm=jnp.sqrt((g_mu_prior ** 2).sum()),
        grad_tau_data_norm=jnp.sqrt((g_tau_data ** 2).sum()),
        grad_tau_prior_norm=jnp.sqrt((g_tau_prior ** 2).sum()),
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
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
    lambda_y: float = 1.0,
    gamma_mu_hidden=None,
    gamma_mu_output=None,
):
    """
    One full M-step for the base BPCN (L_hidden hidden layers + 1 output).

    `data_scale`, `prior_scale` -- see update_layer.

    Output-layer update branches on `net.output_likelihood`:
    - "gaussian"    -> closed-form Eqs. 81-82 via `update_layer(...)` with
                       (y_mean, y_var) as the frozen target.
    - "categorical" -> jax.grad-based update on lambda_y * F_out + gamma_y
                       * KL(q(W_y)||p(W_y)), where F_out is the MEAN or MC
                       softmax NLL. The hidden layer's `update_layer(...)` call 
                       is unchanged in either mode.
    """
    L_hidden = net.L_hidden
    output = net.layers[-1]
    del alpha_hidden, alpha_output  

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
            alpha=net.layers[l].alpha, gamma=gamma_hidden,
            eta_mu=eta_mu_hidden, eta_tau=eta_tau_hidden,
            data_scale=data_scale, prior_scale=prior_scale,
            gamma_mu_override=gamma_mu_hidden,
        )
        new_hiddens.append(new_layer)
        # Per-layer diagnostics key: "hidden_l" for every layer (uniform
        # indexing across all L_hidden values).
        hidden_diags[f"hidden_{l}"] = diag

    # Output-layer update. Presynaptic feature for the output uses psi_L on the *top* hidden latent.
    if net.output_likelihood == "gaussian":
        # Output layer target = (y_mean, y_var) [Gaussian-logit; assumption I3].
        M_out, V_out = psi_moments(net.activations[-1], frozen.m_zs[-1], frozen.v_zs[-1])
        new_output, out_diags = update_layer(
            output,
            M=M_out, V=V_out,
            m_z=y_mean, v_z=y_var,
            alpha=output.alpha, gamma=gamma_output,
            eta_mu=eta_mu_output, eta_tau=eta_tau_output,
            data_scale=data_scale, prior_scale=prior_scale,
            gamma_mu_override=gamma_mu_output,
        )
    elif net.output_likelihood == "categorical":
        # Categorical softmax head.
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
            alpha=output.alpha,
            gamma=gamma_output,
            eta_mu=eta_mu_output,
            eta_tau=eta_tau_output,
            data_scale=data_scale,
            prior_scale=prior_scale,
            lambda_y=lambda_y,
            psi=net.activations[-1],
            gamma_mu_override=gamma_mu_output,
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
