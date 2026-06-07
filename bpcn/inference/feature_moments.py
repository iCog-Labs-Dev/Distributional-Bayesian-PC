"""Moment propagation through deterministic feature map psi.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 4.5 lists four methods for nonlinear psi:
    1. analytic propagation for simple elementwise nonlinearities,
    2. first-order delta approximation,
    3. unscented / sigma-point approximation,
    4. Monte Carlo samples from q^*(z).
- Eq. 93: (m_h, v_h) <- MomentTransform_psi(m_z, v_z).
- Eq. 94: deterministic input -> m_h^0 = psi_0(x_n), v_h^0 = 0.

The active choice is selected at network-construction time via the
`Network.psi` field (see `bpcn/models/network.py`) and dispatched through
`psi_moments` below.
"""
import jax
import jax.numpy as jnp


def identity_moments(m_z, v_z):
    """psi(z) = z: pass-through (Section 4.5 simplest case)."""
    return m_z, v_z


def relu_delta_moments(m_z, v_z):
    """Delta-method approximation for psi(z) = relu(z) (Section 4.5 option 2).

    Mean: relu(m_z).
    Variance: indicator(m_z > 0) * v_z  (Jacobian-squared * input variance).
    """
    mask = (m_z > 0).astype(m_z.dtype)
    return mask * m_z, mask * v_z


_LEAKY_RELU_ALPHA = 0.01  # standard leaky-ReLU negative slope; not exposed as CLI


def leaky_relu_delta_moments(m_z, v_z, alpha: float = _LEAKY_RELU_ALPHA):
    """Delta-method approximation for psi(z) = leaky_relu(z; alpha) (Section 4.5 option 2).

    f(z) = z if z > 0 else alpha * z
    Jacobian:    f'(z) = 1 if z > 0 else alpha
    Delta-method: m_h ~= f(m_z); v_h ~= (f'(m_z))^2 * v_z.

    With mask = (m_z > 0) and coef = mask + alpha * (1 - mask):
        m_h = coef * m_z,    v_h = coef^2 * v_z.
    Recovers `relu_delta_moments` exactly at alpha = 0.
    """
    mask = (m_z > 0).astype(m_z.dtype)
    coef = mask + float(alpha) * (1.0 - mask)
    return coef * m_z, (coef * coef) * v_z


def tanh_delta_moments(m_z, v_z):
    """Delta-method approximation for psi(z) = tanh(z) (Section 4.5 option 2).

    f(z) = tanh(z);   f'(z) = 1 - tanh^2(z) = sech^2(z).
    Delta-method: m_h ~= tanh(m_z); v_h ~= (1 - tanh^2(m_z))^2 * v_z.
    Approximation is tight near m_z = 0 and degrades as |m_z| -> infinity
    (saturation flattens f, but the delta method also predicts vanishing
    variance there, so the two effects partially cancel).
    """
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
    """Apply the exact non-linearity to a sampled latent z.

    Used by MC predictives (`_mc_predict_from_frozen`) and MC categorical
    losses (`mc_categorical_loss`) where we have actual reparametrized
    samples and want the exact transform, not the delta-method moments.

    Consistent with continuation Eq. 26: `a^(s) = W_y^(s) * psi_L(z^(L,(s)))`.
    """
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
    """Dispatch (m_z, v_z) through the chosen psi (Eq. 93).

    Parameters
    ----------
    psi : str
        Feature-map name. One of "identity" (Section 4.5 simplest case),
        "relu", "leaky_relu", or "tanh" (Section 4.5 option 2, delta-method
        approximations).
    m_z, v_z : jax.Array
        Latent posterior moments, same shape.

    Returns
    -------
    (m_h, v_h) : the post-psi presynaptic feature moments fed into the
        next layer's `moment_forward`.
    """
    try:
        return _PSI_DISPATCH[psi](m_z, v_z)
    except KeyError:
        raise ValueError(
            f"unknown psi: {psi!r}; choices: {sorted(_PSI_DISPATCH)}"
        )
