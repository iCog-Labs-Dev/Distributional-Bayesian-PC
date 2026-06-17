"""Inclusion-KL distributional matching loss components."""

from typing import NamedTuple
import jax
import jax.numpy as jnp

from bpcn.utils.safe_math import floor_v


class DistKL(NamedTuple):
    kl: jax.Array       # [B, d_l]
    e:  jax.Array       # [B, d_l]   mean error  (Eq. 66)
    r:  jax.Array       # [B, d_l]   variance residual (Eq. 68)


def gaussian_kl(m_z, v_z, m_p, v_p) -> DistKL:
    """Per-unit Gaussian KL and derived errors for distributional matching."""
    v_p_safe = floor_v(v_p)
    v_z_safe = floor_v(v_z)
    e = m_z - m_p
    r = v_z + e * e - v_p                                  # Eq. 68
    kl = 0.5 * (jnp.log(v_p_safe) - jnp.log(v_z_safe)
                + (v_z + e * e) / v_p_safe
                - 1.0)
    return DistKL(kl=kl, e=e, r=r)
