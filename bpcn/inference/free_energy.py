"""Latent inference free energy F_z used by the E-step.

References (write-up: distributional_predictive_coding_v2.pdf):
- Eq. 40: F_z(lambda_B; phi) = -(N/B) Sum_n E_{q(W) q(z)}[ log p(y|z^L,W_y)
                                                          + Sum_l log p(z^l|z^{l-1},W_l) ]
                              -(N/B) Sum_n H(q_lambda_n)
- Eq. 36: per-transition negative log density (Gaussian).
- Eq. 23: linear-in-weights Gaussian transition z^l ~ N(W_l h, B_l^{-1}).
- psi (Section 4.5): the presynaptic feature for layer l is
  h^{l-1} = psi_{l-1}(z^{l-1}); moments propagate via `psi_moments` dispatch
  (`bpcn/inference/feature_moments.py`). For the base BPCN with L_hidden=1
  the only psi that matters is between z^1 and the output; the input z^0=x
  is clamped (h^0 = x, v_h^0 = 0). Network.psi selects "identity" or "relu".
- Assumption I3 (plan): output y treated as a Gaussian-logit observation; the E-step
  uses the EXACT Gaussian likelihood log p(y | z^L, W_y) = log N(y; W_y z^L, B_y^{-1}),
  ignoring the small Gaussian-target variance epsilon_y (which is only used in the
  M-step distributional matching on W_y).

For base BPCN: L_hidden = 1.  Single hidden layer z^1 in R^{d_1}.

Closed-form expected log density of a Gaussian transition (used below):
    E_{q(W) q(z^l) q(z^{l-1})} [ log N(z^l; W_l h, B_l^{-1}) ]
        = -0.5 * sum_i [ beta_{l,i} * ( (m^l_i - W_l_mean_i * m_h)^2  # squared mean error
                                       + v^l_i                         # latent posterior variance
                                       + Var_q(W_l_i * h) )            # epistemic + propagated variance
                       ] + 0.5 * sum_i log(beta_{l,i}) - constants.

Var_q(W_l_i * h) corresponds exactly to (v_pred_i - beta_{l,i}^{-1}) from Eq. 61.
We therefore reuse moment_forward (models/moments.py).
"""
import jax
import jax.numpy as jnp

from ..models.network import Network
from ..models.moments import moment_forward
from ..utils.safe_math import EPS_V
from .feature_moments import psi_moments
from .categorical_output import categorical_output_loss


_LOG_2PI = float(jnp.log(2.0 * jnp.pi))


def transition_neg_log_density(m_l, v_l, m_h, v_h, layer):
    """Per-example negative expected log density of a Gaussian transition.

    Computes -E[log p(z^l | h, W_l)] = -E[log N(z^l; W_l h, B_l^{-1})] under
    q(W_l) q(z^l) q(z^{l-1}). The 'h' moments are the post-psi (presynaptic)
    moments of the previous layer.

    Parameters
    ----------
    m_l, v_l : [B, d_l]   latent posterior mean and variance for layer l.
    m_h, v_h : [B, p_l]   presynaptic feature mean and variance.
    layer    : Layer      Bayesian layer W_l.

    Returns
    -------
    nll : [B]   per-example value of -E[log p(z^l|h,W_l)].
    """
    m_pred, v_pred = moment_forward(layer, m_h, v_h)              # Eqs. 60-61
    beta = 1.0 / jnp.maximum(layer.beta_inv, EPS_V)               # [d_l]
    # Var_q(W h)_i  =  v_pred_i - beta_inv_i  (Eq. 87 minus residual term)
    var_W_h = v_pred - layer.beta_inv[None, :]
    sq_err = (m_l - m_pred) ** 2                                  # [B, d_l]
    # -E[log N] = 0.5 * beta * ( (m-m_p)^2 + v + Var_W_h ) - 0.5 log(beta) + 0.5 log(2pi)
    quad = 0.5 * (beta[None, :] * (sq_err + v_l + var_W_h)).sum(axis=-1)
    log_det = -0.5 * jnp.log(beta).sum()
    const = 0.5 * layer.d_out * _LOG_2PI
    return quad + log_det + const


def output_neg_log_density(y, m_z, v_z, layer):
    """Per-example -E[log p(y | z^L, W_y)] where y is observed deterministically.

    Identical structure to a transition with target = observed y and target variance = 0.
    Reuses moment_forward on the output layer with (m_z, v_z) as the 'h' moments
    (because psi_L = identity for the base).
    """
    m_pred, v_pred = moment_forward(layer, m_z, v_z)              # Eqs. 60-61, output layer
    beta = 1.0 / jnp.maximum(layer.beta_inv, EPS_V)               # [C]
    var_W_z = v_pred - layer.beta_inv[None, :]
    sq_err = (y - m_pred) ** 2                                    # y has no posterior variance
    quad = 0.5 * (beta[None, :] * (sq_err + var_W_z)).sum(axis=-1)
    log_det = -0.5 * jnp.log(beta).sum()
    const = 0.5 * layer.d_out * _LOG_2PI
    return quad + log_det + const


def latent_neg_entropy(v_l):
    """-H(q_lambda(z^l)) for a diagonal Gaussian = -0.5 sum log(v_i) - const.

    Constants (0.5 d (1 + log 2pi)) are dropped because they do not depend on v_l;
    they only shift F_z by a constant and do not affect E-step gradients or the
    sign of F_z decrease.
    """
    return -0.5 * jnp.log(v_l).sum(axis=-1)


def free_energy(
    net: Network, x, y, m_l, v_l, *,
    output_weight: float = 1.0,
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
):
    """Latent inference free energy F_z (v2 Eq. 40) for L_hidden >= 1.

    For multi-layer, each hidden transition contributes its own NLL term
    (v2 Eq. 36 per layer; the joint factorization is v2 Eq. 17), and the
    latent entropy term sums over all hidden layers. The output boundary
    operates on the *top* hidden latent only (continuation Eq. 38). This
    function is the kappa=0 limit of `shared_free_energy`'s hidden term
    (a per-layer expected NLL) plus the output likelihood; the categorical
    output-head replaces the Gaussian output likelihood per the continuation
    note.

    Parameters
    ----------
    m_l, v_l : tuple of arrays OR single array
        Either a tuple of length L_hidden (multi-layer), or a single array
        (legacy L=1 callers). Each element shape is `[B, d_l]`.
    output_weight : float
        Multiplier on the output likelihood term. 1.0 -> training; 0.0 ->
        test-time target-free inference (Section 6.6). Under the categorical
        head this also plays the role of `lambda_y` (continuation Eq. 2/48).
    y_idx, key, mc_samples_train
        Categorical-mode kwargs. Ignored when
        `net.output_likelihood == "gaussian"`.

    Returns
    -------
    F : scalar               batch-mean F_z (N/B scaling deferred to caller).
    """
    m_zs = m_l if isinstance(m_l, tuple) else (m_l,)
    v_zs = v_l if isinstance(v_l, tuple) else (v_l,)
    dtype = m_zs[0].dtype

    # Per-layer hidden transition expected-NLL + entropy (v2 Eq. 40 hidden term
    # generalized to L_hidden layers; the joint factorization is v2 Eq. 17).
    nll_sum = jnp.zeros((), dtype=dtype)
    neg_H_sum = jnp.zeros((), dtype=dtype)
    for l in range(net.L_hidden):
        if l == 0:
            m_h, v_h = x, jnp.zeros_like(x)
        else:
            m_h, v_h = psi_moments(net.activations[l - 1], m_zs[l - 1], v_zs[l - 1])
        nll_sum = nll_sum + transition_neg_log_density(
            m_zs[l], v_zs[l], m_h=m_h, v_h=v_h, layer=net.layers[l],
        ).mean()
        neg_H_sum = neg_H_sum + latent_neg_entropy(v_zs[l]).mean()

    # Output boundary on the top latent only (continuation Eq. 38).
    m_top, v_top = m_zs[-1], v_zs[-1]
    output = net.layers[-1]
    if net.output_likelihood == "gaussian":
        # Presynaptic feature h^L = psi_L(z^L); moments via `psi_moments`
        # (v2 Eq. 21 / continuation Section 4).
        m_h_out, v_h_out = psi_moments(net.activations[-1], m_top, v_top)
        f_out_per_example = output_neg_log_density(y, m_z=m_h_out, v_z=v_h_out, layer=output)
        return nll_sum + output_weight * f_out_per_example.mean() + neg_H_sum
    elif net.output_likelihood == "categorical":
        f_out_mean = categorical_output_loss(net, m_top, v_top, y_idx, key, mc_samples_train)
        return nll_sum + output_weight * f_out_mean + neg_H_sum
    else:
        raise ValueError(
            f"unknown output_likelihood: {net.output_likelihood!r}; "
            f"choices: 'gaussian', 'categorical'"
        )
