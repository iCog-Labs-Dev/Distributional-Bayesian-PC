"""BPCN linear-in-weights Gaussian layer."""

from typing import NamedTuple
import jax
import jax.numpy as jnp

from bpcn.utils.safe_math import TAU_MIN, TAU_MAX


class Layer(NamedTuple):
    """ Bayesian linear layer with diagonal Gaussian weight posterior. """
    mu: jax.Array
    tau: jax.Array
    beta_inv: jax.Array
    alpha: float

    @property
    def d_out(self) -> int:
        return int(self.mu.shape[0])

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
    """Initialize a layer."""
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
