"""Latent inference free energy F_z used by the E-step.

References (write-up: distributional_predictive_coding_v2.pdf):
- Eq. 40: F_z(lambda_B; phi) = -(N/B) Sum_n E_{q(W) q(z)}[ log p(y|z^L,W_y)
                                                          + Sum_l log p(z^l|z^{l-1},W_l) ]
                              -(N/B) Sum_n H(q_lambda_n)
- Eq. 36: per-transition negative log density (Gaussian).
- Eq. 23: linear-in-weights Gaussian transition z^l ~ N(W_l h, B_l^{-1}).
- Assumption I2 (plan): psi = identity, so h^{l-1} = z^{l-1} and v_h^{l-1} = v_z^{l-1}.
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


def free_energy(net: Network, x, y, m_l, v_l, *, output_weight: float = 1.0):
    """Latent inference free energy F_z (Eq. 40) for the base BPCN (L_hidden = 1).

    Parameters
    ----------
    net : Network            current weight posterior (treated as stop_gradient by caller).
    x   : [B, d_0]           inputs (z^0).
    y   : [B, C]             observed one-hot logit targets (m_z^{out}).
                              Ignored when output_weight == 0 (Section 6.6 test-time use).
    m_l : [B, d_1]           latent mean of the single hidden layer.
    v_l : [B, d_1]           latent variance.
    output_weight : float    Multiplier on the output likelihood term.
                              1.0 -> training (label observed; Algorithm 1).
                              0.0 -> test-time target-free inference (Section 6.6).

    Returns
    -------
    F : scalar               mean F_z over the batch (N/B scaling deferred to caller).
    """
    assert net.L_hidden == 1, "free_energy here is specialised to L_hidden == 1"
    hidden = net.layers[0]
    output = net.layers[1]

    # Layer 1 transition: presynaptic is the input x with zero variance (Eq. 94).
    nll_1 = transition_neg_log_density(
        m_l, v_l,
        m_h=x, v_h=jnp.zeros_like(x),
        layer=hidden,
    )
    # Output likelihood: presynaptic is the hidden latent with variance v_l (identity psi).
    nll_y = output_neg_log_density(y, m_z=m_l, v_z=v_l, layer=output)
    neg_H = latent_neg_entropy(v_l)
    F_per_example = nll_1 + output_weight * nll_y + neg_H          # [B]
    return F_per_example.mean()
