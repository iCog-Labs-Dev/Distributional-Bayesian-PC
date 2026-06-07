"""Layerwise predictive moments for BPCN.

References (write-up: distributional_predictive_coding_v2.pdf):
- Eqs. 60-61: moment-matched predictive mean and variance.
  m_pred,n,i  = sum_j  mu_{l,ij} * m_{h,n,j}
  v_pred,n,i  = beta_{l,i}^{-1}
              + sum_j [ mu_{l,ij}^2 * v_{h,n,j}
                      + sigma_{l,ij}^2 * (m_{h,n,j}^2 + v_{h,n,j}) ]

- Eq. 87 (Section 4.9): variance decomposition into residual + propagated latent + epistemic weight.
"""
from typing import Tuple
import jax

from .layer import Layer


def moment_forward(
    layer: Layer,
    m_h: jax.Array,
    v_h: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Predictive (mean, variance) for one BPCN layer (Eqs. 60-61).

    Parameters
    ----------
    layer : Layer
        Bayesian layer with mu [d_out, p_in], tau [d_out, p_in], beta_inv [d_out].
    m_h : jax.Array, shape [B, p_in]
        Presynaptic mean (m_h^{l-1}).
    v_h : jax.Array, shape [B, p_in]
        Presynaptic variance (v_h^{l-1}).

    Returns
    -------
    m_pred : [B, d_out]
    v_pred : [B, d_out]
    """
    sigma2 = layer.sigma2()                                # [d_out, p_in]
    m_pred = m_h @ layer.mu.T                              # [B, d_out]  (Eq. 60)
    # v_pred = beta_inv + sum_j [ mu_ij^2 * v_h,j + sigma_ij^2 * (m_h,j^2 + v_h,j) ]
    propagated = v_h @ (layer.mu ** 2).T                   # mu^2 * v_h  -> [B, d_out]
    epistemic  = (m_h ** 2 + v_h) @ sigma2.T               # sigma^2 * (m^2+v)
    v_pred = layer.beta_inv[None, :] + propagated + epistemic  # [B, d_out]   (Eq. 61)
    return m_pred, v_pred


def variance_components(
    layer: Layer,
    m_h: jax.Array,
    v_h: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Decompose v_pred into the three terms of Eq. 87.

    Returns
    -------
    residual : [d_out]                broadcast over batch; this is beta_inv only.
    propagated : [B, d_out]           sum_j mu_ij^2 * v_h,j
    epistemic : [B, d_out]            sum_j sigma_ij^2 * (m_h,j^2 + v_h,j)
    """
    sigma2 = layer.sigma2()
    propagated = v_h @ (layer.mu ** 2).T
    epistemic  = (m_h ** 2 + v_h) @ sigma2.T
    residual   = layer.beta_inv                            # [d_out]
    return residual, propagated, epistemic
