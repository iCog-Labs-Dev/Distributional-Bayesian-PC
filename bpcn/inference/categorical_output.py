""" Bayesian categorical (softmax) output head for the shared-energy DBPCN. """

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from bpcn.models.layer import Layer
from bpcn.losses.weight_kl import gaussian_weight_kl, gaussian_weight_kl_components
from bpcn.utils.safe_math import clamp_tau
from bpcn.inference.feature_moments import psi_moments, apply_psi_sample


def _mean_logits(mu, m_h):
    """Posterior-mean logits a^mean = mu_y @ m_h^T (Eq. 20)."""
    return m_h @ mu.T


def _per_example_nll(logits, y_idx):
    """Per-example -log softmax(logits)_{y_idx}.  [B] for logits [B, C]."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    B = y_idx.shape[0]
    return -log_probs[jnp.arange(B), y_idx]


def mean_categorical_loss(net, m_z, v_z, y_idx):
    """
    mean averaged over the batch.

    Pure function of (output.mu, psi, m_z, y_idx). Note v_z is ignored
    under the pure-mean estimator (Eq. 23 has no v_z dependence). The
    return is in per-data-point ("nats per example") scale, matching the
    codebase's data-term convention (mean over the batch).
    """
    m_h, _ = psi_moments(net.activations[-1], m_z, v_z)
    output = net.layers[-1]
    logits = _mean_logits(output.mu, m_h)
    return _per_example_nll(logits, y_idx).mean()


def mc_categorical_loss(net, m_z, v_z, y_idx, key, S):
    """
    MC averaged over the batch and S samples.

    Parameters
    ----------
    key : jax.Array
        PRNG key. The MC estimator needs fresh randomness per call.
    S : int (static) Number of MC samples. Use 1 for BBB-style SGD
    """
    output = net.layers[-1]
    sigma_y = jnp.sqrt(jnp.exp(output.tau))
    sigma_z = jnp.sqrt(v_z)
    psi_top = net.activations[-1]
    mu = output.mu

    def one_sample(k):
        k_W, k_z = jax.random.split(k)
        W_sample = mu + sigma_y * jax.random.normal(k_W, mu.shape)
        z_sample = m_z + sigma_z * jax.random.normal(k_z, m_z.shape)
        z_sample = apply_psi_sample(psi_top, z_sample)
        logits = z_sample @ W_sample.T                 # [B, C]
        return _per_example_nll(logits, y_idx)         # [B]

    keys = jax.random.split(key, S)
    nll_samples = jax.vmap(one_sample)(keys)           # [S, B]
    return nll_samples.mean()                          # per-data-point nats


def categorical_output_loss(net, m_z, v_z, y_idx, key, S):
    """ Dispatcher: pick MEAN or MC categorical loss based on net.output_estimator. """
    if net.output_estimator == "mean":
        return mean_categorical_loss(net, m_z, v_z, y_idx)
    elif net.output_estimator == "mc":
        return mc_categorical_loss(net, m_z, v_z, y_idx, key, S)
    else:
        raise ValueError(
            f"unknown output_estimator: {net.output_estimator!r}; "
            f"choices: 'mean', 'mc'"
        )

class CategoricalOutputDiagnostics(NamedTuple):
    """ Diagnostics for the categorical output M-step. """
    kl_data: jax.Array
    kl_data_before: jax.Array
    kl_data_after: jax.Array
    kl_data_delta: jax.Array
    kl_data_loop_before: jax.Array
    kl_data_loop_after: jax.Array
    kl_data_loop_delta: jax.Array
    kl_weight: jax.Array
    grad_mu_norm: jax.Array
    grad_tau_norm: jax.Array
    grad_mu_data_norm: jax.Array
    grad_mu_prior_norm: jax.Array
    grad_tau_data_norm: jax.Array
    grad_tau_prior_norm: jax.Array
    mean_sigma2: jax.Array
    var_residual_pos_frac: jax.Array
    components: dict


def _categorical_objective(
    mu, tau, m_z, v_z, y_idx, key, S, *,
    estimator, psi, alpha, gamma, lambda_y, data_scale, prior_scale, B,
    gamma_mu_override=None,
):    
    """The scalar objective J(mu, tau) for the categorical output M-step."""
    if estimator == "mean":
        m_h, _ = psi_moments(psi, m_z, v_z)
        logits = m_h @ mu.T
        loss = _per_example_nll(logits, y_idx).mean()
    else:  # "mc"
        sigma_y = jnp.sqrt(jnp.exp(tau))
        sigma_z = jnp.sqrt(v_z)

        def one_sample(k):
            k_W, k_z = jax.random.split(k)
            W_sample = mu + sigma_y * jax.random.normal(k_W, mu.shape)
            z_sample = m_z + sigma_z * jax.random.normal(k_z, m_z.shape)
            z_sample = apply_psi_sample(psi, z_sample)
            logits = z_sample @ W_sample.T
            return _per_example_nll(logits, y_idx)

        keys = jax.random.split(key, S)
        loss = jax.vmap(one_sample)(keys).mean()

    data_term = lambda_y * data_scale * B * loss
    if gamma_mu_override is None:
        prior_term = prior_scale * gamma * gaussian_weight_kl(mu, tau, alpha).sum()
    else:
        mu_term, var_term = gaussian_weight_kl_components(mu, tau, alpha)
        prior_term = prior_scale * (
            gamma_mu_override * mu_term.sum() + gamma * var_term.sum()
        )
    return data_term + prior_term, loss


def categorical_output_update(
    output: Layer,
    frozen,            # FrozenLatents-like with .m_z, .v_z
    y_idx: jax.Array,
    key: jax.Array,
    *,
    estimator: str,
    S: int,
    alpha: float,
    gamma: float,
    eta_mu: float,
    eta_tau: float,
    data_scale: float,
    prior_scale: float,
    lambda_y: float,
    psi: str,
    gamma_mu_override=None,
) -> Tuple[Layer, CategoricalOutputDiagnostics]:
    """One M-step update on the categorical output head (Eq. 22), returning the new Layer and diagnostics."""
    
    m_z = jax.lax.stop_gradient(frozen.m_z)
    v_z = jax.lax.stop_gradient(frozen.v_z)
    y_idx_sg = jax.lax.stop_gradient(y_idx)
    B = int(y_idx.shape[0])

    def J_with_loss(mu, tau):
        return _categorical_objective(
            mu, tau, m_z, v_z, y_idx_sg, key, S,
            estimator=estimator, psi=psi,
            alpha=alpha, gamma=gamma, lambda_y=lambda_y,
            data_scale=data_scale, prior_scale=prior_scale, B=B,
            gamma_mu_override=gamma_mu_override,
        )

    (_, loss_before), (g_mu, g_tau) = jax.value_and_grad(
        J_with_loss, argnums=(0, 1), has_aux=True,
    )(output.mu, output.tau)

    alpha2 = alpha * alpha
    gamma_mu_eff = gamma if gamma_mu_override is None else gamma_mu_override
    g_mu_prior_vec  = prior_scale * gamma_mu_eff * output.mu / alpha2
    sigma2_old = jnp.exp(output.tau)
    g_tau_prior_vec = prior_scale * 0.5 * gamma * (sigma2_old / alpha2 - 1.0)
    g_mu_data_vec   = g_mu  - g_mu_prior_vec
    g_tau_data_vec  = g_tau - g_tau_prior_vec
    grad_mu_data_norm   = jnp.sqrt((g_mu_data_vec   ** 2).sum())
    grad_mu_prior_norm  = jnp.sqrt((g_mu_prior_vec  ** 2).sum())
    grad_tau_data_norm  = jnp.sqrt((g_tau_data_vec  ** 2).sum())
    grad_tau_prior_norm = jnp.sqrt((g_tau_prior_vec ** 2).sum())

    new_mu = output.mu - eta_mu * g_mu
    new_tau = clamp_tau(output.tau - eta_tau * g_tau)
    new_layer = Layer(
        mu=new_mu, tau=new_tau,
        beta_inv=output.beta_inv, alpha=output.alpha,
    )

    _, loss_after = J_with_loss(new_mu, new_tau)

    sigma2_new = jnp.exp(new_tau)
    kl_weight = gaussian_weight_kl(new_mu, new_tau, alpha).mean()

    components = dict(
        # Logit-space statistics in place of the Gaussian residual/propagated/
        # epistemic breakdown. Useful at-a-glance even though the
        # categorical head has no v_p decomposition.
        residual_mean=jnp.asarray(0.0, dtype=new_mu.dtype),
        propagated_mean=jnp.asarray(0.0, dtype=new_mu.dtype),
        epistemic_mean=jnp.asarray(0.0, dtype=new_mu.dtype),
        v_p_mean=jnp.asarray(0.0, dtype=new_mu.dtype),
        v_p_min=jnp.asarray(0.0, dtype=new_mu.dtype),
        e_abs_mean=jnp.asarray(loss_after, dtype=new_mu.dtype),
        r_abs_mean=jnp.asarray(0.0, dtype=new_mu.dtype),
        v_p_mean_before=jnp.asarray(0.0, dtype=new_mu.dtype),
        v_p_min_before=jnp.asarray(0.0, dtype=new_mu.dtype),
        e_abs_mean_before=jnp.asarray(loss_before, dtype=new_mu.dtype),
        r_abs_mean_before=jnp.asarray(0.0, dtype=new_mu.dtype),
    )
    diags = CategoricalOutputDiagnostics(
        kl_data=loss_after,
        kl_data_before=loss_before,
        kl_data_after=loss_after,
        kl_data_delta=loss_after - loss_before,
        kl_data_loop_before=loss_before,
        kl_data_loop_after=loss_after,
        kl_data_loop_delta=loss_after - loss_before,
        kl_weight=kl_weight,
        grad_mu_norm=jnp.sqrt((g_mu ** 2).sum()),
        grad_tau_norm=jnp.sqrt((g_tau ** 2).sum()),
        grad_mu_data_norm=grad_mu_data_norm,
        grad_mu_prior_norm=grad_mu_prior_norm,
        grad_tau_data_norm=grad_tau_data_norm,
        grad_tau_prior_norm=grad_tau_prior_norm,
        mean_sigma2=sigma2_new.mean(),
        var_residual_pos_frac=jnp.asarray(0.0, dtype=new_mu.dtype),
        components=components,
    )
    return new_layer, diags
