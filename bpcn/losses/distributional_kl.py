"""Inclusion-KL distributional matching loss components.

References (write-up: distributional_predictive_coding_v2.pdf):
- Eq. 63: KL direction is KL(q^*(z) || q_pred(z))  -- inclusion / mass-covering.
- Eq. 65: per-unit Gaussian KL:
    KL(N(m_z, v_z) || N(m_p, v_p))
        = 0.5 [ log(v_p / v_z)  +  (v_z + (m_z - m_p)^2) / v_p  - 1 ].
- Eq. 66: mean error    e_{n,i} = m_{z,n,i} - m_{pred,n,i}.
- Eq. 67: target second central mismatch a_{n,i} = v_{z,n,i} + e_{n,i}^2.
- Eq. 68: variance residual r_{n,i} = a_{n,i} - v_{pred,n,i}.

These quantities feed Eqs. 81-82 in training/m_step.py.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from ..utils.safe_math import floor_v


class DistKL(NamedTuple):
    kl: jax.Array       # [B, d_l]
    e:  jax.Array       # [B, d_l]   mean error  (Eq. 66)
    r:  jax.Array       # [B, d_l]   variance residual (Eq. 68)


def gaussian_kl(m_z, v_z, m_p, v_p) -> DistKL:
    """Per-unit Gaussian KL (Eq. 65) and derived errors (Eqs. 66, 68).

    Always evaluated in the inclusion-KL direction KL(q^* || q_pred) (Eq. 63).
    """
    v_p_safe = floor_v(v_p)
    v_z_safe = floor_v(v_z)
    e = m_z - m_p
    r = v_z + e * e - v_p                                  # Eq. 68
    kl = 0.5 * (jnp.log(v_p_safe) - jnp.log(v_z_safe)
                + (v_z + e * e) / v_p_safe
                - 1.0)
    return DistKL(kl=kl, e=e, r=r)
