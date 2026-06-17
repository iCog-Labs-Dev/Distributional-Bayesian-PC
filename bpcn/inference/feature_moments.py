""" Moment propagation through deterministic feature map psi. """

import jax
import jax.numpy as jnp


def identity_moments(m_z, v_z):
    """psi(z) = z: pass-through (Section 4.5 simplest case)."""
    return m_z, v_z


def relu_delta_moments(m_z, v_z):
    """Delta-method approximation for psi(z) = relu(z)."""
    mask = (m_z > 0).astype(m_z.dtype)
    return mask * m_z, mask * v_z


_LEAKY_RELU_ALPHA = 0.01  # standard leaky-ReLU negative slope; not exposed as CLI


def leaky_relu_delta_moments(m_z, v_z, alpha: float = _LEAKY_RELU_ALPHA):
    """Delta-method approximation for psi(z) = leaky_relu(z; alpha),"""
    mask = (m_z > 0).astype(m_z.dtype)
    coef = mask + float(alpha) * (1.0 - mask)
    return coef * m_z, (coef * coef) * v_z


def tanh_delta_moments(m_z, v_z):
    """Delta-method approximation for psi(z) = tanh(z)."""
    t = jnp.tanh(m_z)
    jac = 1.0 - t * t
    return t, (jac * jac) * v_z


_PSI_DISPATCH = {
    "identity": identity_moments,
    "relu": relu_delta_moments,
    "leaky_relu": leaky_relu_delta_moments,
    "tanh": tanh_delta_moments,
}


def apply_psi_sample(psi: str, z):
    """Apply the exact non-linearity to a sampled latent z."""
    if psi == "identity":
        return z
    if psi == "relu":
        return jax.nn.relu(z)
    if psi == "leaky_relu":
        return jax.nn.leaky_relu(z, negative_slope=_LEAKY_RELU_ALPHA)
    if psi == "tanh":
        return jnp.tanh(z)
    raise ValueError(
        f"unknown psi: {psi!r}; choices: {sorted(_PSI_DISPATCH)}"
    )


def psi_moments(psi: str, m_z, v_z):
    """Dispatch (m_z, v_z) through the chosen psi."""
    try:
        return _PSI_DISPATCH[psi](m_z, v_z)
    except KeyError:
        raise ValueError(
            f"unknown psi: {psi!r}; choices: {sorted(_PSI_DISPATCH)}"
        )
