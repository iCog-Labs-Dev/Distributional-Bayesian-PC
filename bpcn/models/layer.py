"""BPCN linear-in-weights Gaussian layer.

References (write-up: distributional_predictive_coding_v2.pdf):
- Eq. 22: weight matrix W_l in R^{d_l x p_l}.
- Eq. 23: linear-in-weights Gaussian transition z^l ~ N(W_l h^{l-1}, B_l^{-1}).
- Eq. 24: diagonal residual precision B_l = diag(beta_{l,i}).
- Eq. 27: zero-mean independent Gaussian prior p(W_l) = prod N(0, alpha_l^2).
- Eq. 28: diagonal Gaussian posterior q_phi_l(W_l) = prod N(mu, sigma^2).
- Eq. 33: log-variance tau_{ij} = log sigma^2_{ij}.
- Eqs. 98-99: initialization mu ~ N(0, kappa^2), tau <- log sigma_0^2.
- Eq. 111: sigma^2 = exp(tau).
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from ..utils.safe_math import TAU_MIN, TAU_MAX


class Layer(NamedTuple):
    """Bayesian linear layer with diagonal Gaussian weight posterior.

    Attributes
    ----------
    mu : jax.Array, shape [d_out, p_in]
        Weight posterior mean (Eq. 28).
    tau : jax.Array, shape [d_out, p_in]
        Weight posterior log-variance, sigma^2 = exp(tau) (Eqs. 33, 111).
    beta_inv : jax.Array, shape [d_out]
        Residual variance B_l^{-1} (Eq. 24). Held fixed in the base (A4).
    alpha : float
        Prior scale alpha_l (Eq. 27). Scalar shared within the layer.
    """
    mu: jax.Array
    tau: jax.Array
    beta_inv: jax.Array
    alpha: float

    @property
    def d_out(self) -> int:
        return int(self.mu.shape[0])

    @property
    def p_in(self) -> int:
        return int(self.mu.shape[1])

    def sigma2(self) -> jax.Array:
        """Weight posterior variance sigma^2 = exp(tau) (Eq. 111)."""
        return jnp.exp(self.tau)


def init_layer(
    key: jax.Array,
    d_out: int,
    p_in: int,
    *,
    alpha: float = 1.0,
    beta_inv: float = 1e-2,
    init_kind: str = "xavier",
    init_log_var: float = -6.0,
) -> Layer:
    """Initialize a layer per Eqs. 98-99.

    Parameters
    ----------
    key : PRNGKey
    d_out : int
        Output dimensionality d_l.
    p_in : int
        Input (presynaptic feature) dimensionality p_l.
    alpha : float
        Prior scale alpha_l (Eq. 27). Used for KL-to-prior in Eq. 77.
    beta_inv : float or array
        Residual variance B_l^{-1}. Scalar broadcast to [d_out] (Eq. 24, A4).
    init_kind : {"xavier", "he"}
        Scaling rule for kappa_l (Eq. 98).
    init_log_var : float
        Initial tau value (log sigma_0^2) (Eq. 99). Small to start near-deterministic.
    """
    if init_kind == "xavier":
        kappa = jnp.sqrt(1.0 / p_in)
    elif init_kind == "he":
        kappa = jnp.sqrt(2.0 / p_in)
    else:
        raise ValueError(f"unknown init_kind: {init_kind!r}")

    mu = jax.random.normal(key, (d_out, p_in)) * kappa             # Eq. 98
    tau = jnp.full((d_out, p_in), float(init_log_var))             # Eq. 99
    tau = jnp.clip(tau, TAU_MIN, TAU_MAX)
    beta_inv_arr = jnp.full((d_out,), float(beta_inv))
    return Layer(mu=mu, tau=tau, beta_inv=beta_inv_arr, alpha=float(alpha))
