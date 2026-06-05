"""Bayesian categorical (softmax) output head for the shared-energy DBPCN.

References (write-up: categorical_output_dbpcn_continuation.pdf):
- Eq. 1 / Eq. 18 : p(y_n=c | z_n^L, W_y) = softmax(W_y psi_L(z_n^L))_c.
- Eq. 2 / Eq. 48 : F_cat-DPC = lambda_y F_out + F_trans-DPC + F_weight-KL.
- Section 4.1 (Option A, MEAN estimator):
    Eq. 20  m_n^a = mu_y M_n^L            (posterior-mean logits)
    Eq. 21  ell^mean = -log softmax(m_n^a)_{y_n}
    Eq. 22  F_out^mean = (N/|B|) sum ell^mean
    Eq. 23  d ell^mean / d M_n^L = mu_y^T (p_n^mean - y_n^oh)
- Section 4.2 (Option B, MC estimator):
    Eq. 24  W_y^(s) = mu_y + sigma_y * eps^(s)_y
    Eq. 25  z^{L,(s)} = m_z^L + sqrt(v_z^L) * xi^(s)
    Eq. 26  h^{L,(s)} = psi_L(z^{L,(s)}),  a^(s) = W_y^(s) h^{L,(s)}
    Eq. 27  ell^MC = (1/S) sum_s [-log softmax(a^(s))_{y_n}]
    Eq. 28  F_out^MC = (N/|B|) sum ell^MC
- Section 6.2 (output M-step):
    Eq. 43  phi_y <- phi_y - eta_y grad [lambda_y F_out + gamma_y KL(q(W_y)||p(W_y))]
    Eq. 44/45  KL term -- same as Eq. 77 of v2 / Eq. 52 of M-SE extension.

Prediction (Section 8):
- Eq. 47  p_n^mean(c) = softmax(mu_y M_n^L)_c
- Eq. 34  p_hat(y=c|x) = (1/S) sum_s softmax(a^(s))_c
These are already implemented by `_mean_predict_from_frozen` and
`_mc_predict_from_frozen` in `bpcn/evaluation/predict.py` -- they need no
changes to support the categorical head.

This module supplies:
- `mean_categorical_loss(net, m_z, v_z, y_idx) -> scalar`       (Eq. 21/22)
- `mc_categorical_loss(net, m_z, v_z, y_idx, key, S) -> scalar` (Eq. 27/28)
- `categorical_output_loss(net, m_z, v_z, y_idx, key, S) -> scalar`
  dispatcher used by `shared_free_energy` / `free_energy` to compute F_out.
- `categorical_output_update(...)` used by the M-step in place of
  `update_layer(...)` for the output slot. It builds a scalar objective
  J(mu_y, tau_y) = lambda_y * data_scale * B * loss(mu, tau)
                 + prior_scale * gamma_y * sum KL(q(W_{y,cj})||p(W_{y,cj}))
  and applies one gradient-descent step via `jax.grad`. The data_scale * B
  factor keeps eta values in the same 1e-3 ballpark as the Gaussian-mode
  update_layer (see `bpcn/training/m_step.py` SCALING NOTE).

TODO (Eq. 30-32 of the continuation): local-reparameterization variant
that propagates logit moments (m_n^a, v_n^a) and samples logits directly
(lower variance, ignores cross-class covariance). Deferred; the per-sample
reparam form of Eqs. 24-27 ships here.
"""
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from ..models.layer import Layer
from ..losses.weight_kl import gaussian_weight_kl
from ..utils.safe_math import clamp_tau
from .feature_moments import psi_moments, apply_psi_sample


def _mean_logits(mu, m_h):
    """Posterior-mean logits a^mean = mu_y @ m_h^T (Eq. 20)."""
    return m_h @ mu.T


def _per_example_nll(logits, y_idx):
    """Per-example -log softmax(logits)_{y_idx}.  [B] for logits [B, C]."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    B = y_idx.shape[0]
    return -log_probs[jnp.arange(B), y_idx]


def mean_categorical_loss(net, m_z, v_z, y_idx):
    """ell^mean averaged over the batch (Eq. 21-22 of the continuation).

    Pure function of (output.mu, psi, m_z, y_idx). Note v_z is ignored
    under the pure-mean estimator (Eq. 23 has no v_z dependence). The
    return is in per-data-point ("nats per example") scale, matching the
    codebase's data-term convention (mean over the batch).
    """
    # Output head consumes h^L = psi_L(z^L); psi_L is the last entry of the
    # per-layer activations tuple (continuation Section 4 / Eq. 16).
    m_h, _ = psi_moments(net.activations[-1], m_z, v_z)
    output = net.layers[-1]
    logits = _mean_logits(output.mu, m_h)
    return _per_example_nll(logits, y_idx).mean()


def mc_categorical_loss(net, m_z, v_z, y_idx, key, S):
    """ell^MC averaged over the batch and S samples (Eq. 27-28 of continuation).

    Reparametrize W_y^(s) = mu_y + sigma_y * eps_W^(s) and
    z^{L,(s)} = m_z + sqrt(v_z) * eps_z^(s); apply psi_L to z^{L,(s)} and
    matmul to obtain logits a^(s) = W_y^(s) psi_L(z^{L,(s)}). The
    per-sample, per-example NLL is -log softmax(a^(s))_{y_n}; the loss
    is averaged over both S and B (per-data-point scale).

    The ReLU-on-sample form is exact for the sample (not the delta-method
    moment approximation), consistent with how `_mc_predict_from_frozen`
    handles psi in `bpcn/evaluation/predict.py`.

    Parameters
    ----------
    key : jax.Array
        PRNG key. The MC estimator needs fresh randomness per call; the
        caller is responsible for splitting/folding.
    S : int (static)
        Number of MC samples. Use 1 for BBB-style SGD; larger S reduces
        gradient variance at proportional cost.
    """
    output = net.layers[-1]
    sigma_y = jnp.sqrt(jnp.exp(output.tau))
    sigma_z = jnp.sqrt(v_z)
    # Output head boundary: only psi_L (continuation Eq. 26) applies to the
    # sampled top latent; interior activations are baked into m_z via the E-step.
    psi_top = net.activations[-1]
    mu = output.mu

    def one_sample(k):
        k_W, k_z = jax.random.split(k)
        W_sample = mu + sigma_y * jax.random.normal(k_W, mu.shape)
        z_sample = m_z + sigma_z * jax.random.normal(k_z, m_z.shape)
        # Apply psi_L exactly to the sample (Section 4.5 option 4 / matches
        # _mc_predict_from_frozen). `apply_psi_sample` dispatches across
        # identity / relu / leaky_relu / tanh.
        z_sample = apply_psi_sample(psi_top, z_sample)
        logits = z_sample @ W_sample.T                 # [B, C]
        return _per_example_nll(logits, y_idx)         # [B]

    keys = jax.random.split(key, S)
    nll_samples = jax.vmap(one_sample)(keys)           # [S, B]
    return nll_samples.mean()                          # per-data-point nats


def categorical_output_loss(net, m_z, v_z, y_idx, key, S):
    """Dispatcher: pick MEAN or MC categorical loss based on net.output_estimator.

    Used by `shared_free_energy` and `free_energy` to compute F_out when
    `net.output_likelihood == "categorical"`. The `key`/`S` arguments are
    accepted unconditionally so the dispatch is shape-stable; MEAN ignores
    them.
    """
    if net.output_estimator == "mean":
        return mean_categorical_loss(net, m_z, v_z, y_idx)
    elif net.output_estimator == "mc":
        return mc_categorical_loss(net, m_z, v_z, y_idx, key, S)
    else:
        raise ValueError(
            f"unknown output_estimator: {net.output_estimator!r}; "
            f"choices: 'mean', 'mc'"
        )


# ---------------------------------------------------------------------------
# Output M-step (replaces update_layer for the output slot under categorical mode)
# ---------------------------------------------------------------------------


class CategoricalOutputDiagnostics(NamedTuple):
    """Diagnostics for the categorical output M-step.

    Shape-compatible with the keys consumed by `_with_loop_kl_summary` and
    EpochDiagnostics (kl_data_*, kl_weight, grad_*_norm, mean_sigma2,
    var_residual_pos_frac, components). Gaussian-only fields are filled
    with stub values:
    - `kl_data_*` carry the per-example categorical NLL (data term) so the
      loop's "kl_data" trace remains a meaningful "did the data term go
      down" signal. Always positive (cross-entropy).
    - `var_residual_pos_frac` is 0.0 (no variance residual in categorical
      mode).
    - `components` carries logit-space statistics so the variance
      decomposition diagnostic remains populated.
    """
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
    mean_sigma2: jax.Array
    var_residual_pos_frac: jax.Array
    components: dict


def _categorical_objective(
    mu, tau, m_z, v_z, y_idx, key, S, *,
    estimator, psi, alpha, gamma, lambda_y, data_scale, prior_scale, B,
):
    """J(mu, tau) = lambda_y * data_scale * B * loss + prior_scale * gamma * KL_prior.

    The (data_scale * B) factor makes the data-term magnitude match the
    sum-then-scale convention of `update_layer` (see m_step.py SCALING
    NOTE), so eta_mu_output / eta_tau_output stay in the conventional
    1e-3 range across modes.
    """
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
            # Apply psi_L exactly to the sample (Section 4.5 option 4 /
            # continuation Eq. 26: a^(s) = W^(s) psi_L(z^(L,(s)))). Must
            # dispatch across all four activations -- a previous version
            # only handled "relu" and silently became the identity for
            # leaky_relu/tanh, training against an identity-head loss.
            z_sample = apply_psi_sample(psi, z_sample)
            logits = z_sample @ W_sample.T
            return _per_example_nll(logits, y_idx)

        keys = jax.random.split(key, S)
        loss = jax.vmap(one_sample)(keys).mean()

    data_term = lambda_y * data_scale * B * loss
    prior_term = prior_scale * gamma * gaussian_weight_kl(mu, tau, alpha).sum()
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
) -> Tuple[Layer, CategoricalOutputDiagnostics]:
    """One M-step update on the categorical output head (Eq. 43 of continuation).

    Uses `jax.grad` on the scalar objective `J(mu, tau)` rather than a
    closed-form Eqs. 81-82 update, because the categorical likelihood has
    no Gaussian inclusion-KL form -- the closed-form distributional
    projection applies only to Gaussian transitions. The categorical loss
    plus weight-KL is exactly what the continuation note descends in
    Section 6.2.
    """
    m_z = jax.lax.stop_gradient(frozen.m_z)
    v_z = jax.lax.stop_gradient(frozen.v_z)
    y_idx_sg = jax.lax.stop_gradient(y_idx)
    B = int(y_idx.shape[0])

    def J_with_loss(mu, tau):
        # value_and_grad with has_aux=True wants (scalar, aux). We return
        # (J, loss) so the per-example NLL is available for diagnostics
        # without a second forward pass.
        return _categorical_objective(
            mu, tau, m_z, v_z, y_idx_sg, key, S,
            estimator=estimator, psi=psi,
            alpha=alpha, gamma=gamma, lambda_y=lambda_y,
            data_scale=data_scale, prior_scale=prior_scale, B=B,
        )

    (_, loss_before), (g_mu, g_tau) = jax.value_and_grad(
        J_with_loss, argnums=(0, 1), has_aux=True,
    )(output.mu, output.tau)

    new_mu = output.mu - eta_mu * g_mu
    new_tau = clamp_tau(output.tau - eta_tau * g_tau)
    new_layer = Layer(
        mu=new_mu, tau=new_tau,
        beta_inv=output.beta_inv, alpha=output.alpha,
    )

    # Diagnostics: re-evaluate loss after the update so `kl_data_after` is a
    # meaningful "did the data term drop" reading. Under MC, both calls use
    # the SAME `key` so the diagnostic comparison is apples-to-apples.
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
        mean_sigma2=sigma2_new.mean(),
        var_residual_pos_frac=jnp.asarray(0.0, dtype=new_mu.dtype),
        components=components,
    )
    return new_layer, diags
