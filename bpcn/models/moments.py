"""Layerwise predictive moments for BPCN."""

from typing import Tuple
import jax

from bpcn.models.layer import Layer


def moment_forward(
    layer: Layer,
    m_h: jax.Array,
    v_h: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Predictive (mean, variance) for one BPCN layer."""
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
    """Decompose v_pred into the three terms.
    residual : [d_out]                broadcast over batch; this is beta_inv only.
    propagated : [B, d_out]           sum_j mu_ij^2 * v_h,j
    epistemic : [B, d_out]            sum_j sigma_ij^2 * (m_h,j^2 + v_h,j)
    """
    sigma2 = layer.sigma2()
    propagated = v_h @ (layer.mu ** 2).T
    epistemic  = (m_h ** 2 + v_h) @ sigma2.T
    residual   = layer.beta_inv                            # [d_out]
    return residual, propagated, epistemic
