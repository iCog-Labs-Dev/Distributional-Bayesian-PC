"""Moment propagation through deterministic feature map psi.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 4.5 lists four methods for nonlinear psi:
    1. analytic propagation for simple elementwise nonlinearities,
    2. first-order delta approximation,
    3. unscented / sigma-point approximation,
    4. Monte Carlo samples from q^*(z).
- Eq. 93: (m_h, v_h) <- MomentTransform_psi(m_z, v_z).
- Eq. 94: deterministic input -> m_h^0 = psi_0(x_n), v_h^0 = 0.

Base: psi = identity (assumption I2). MomentTransform is the identity map.
"""
from typing import Callable, Tuple
import jax.numpy as jnp


def identity_moments(m_z, v_z):
    """psi(z) = z: pass-through (Section 4.5 simplest case)."""
    return m_z, v_z


def relu_delta_moments(m_z, v_z):
    """Delta-method approximation for psi(z) = relu(z) (Section 4.5 option 2).

    Mean: relu(m_z).
    Variance: indicator(m_z > 0) * v_z  (Jacobian-squared * input variance).

    Kept for sub-milestone M-5; not used in the base linear stage.
    """
    mask = (m_z > 0).astype(m_z.dtype)
    return mask * m_z, mask * v_z


def zero_input_moments(x):
    """For the first layer: m_h^0 = x, v_h^0 = 0 (Eq. 94 with deterministic input)."""
    return x, jnp.zeros_like(x)
