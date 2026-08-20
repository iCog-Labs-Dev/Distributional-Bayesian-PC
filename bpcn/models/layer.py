"""Bayesian linear layer used by BPCN."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bpcn.utils.safe_math import clamp_weight_log_variance


class Layer(NamedTuple):
    """Linear layer with a diagonal-Gaussian weight posterior."""

    mean: jax.Array
    log_variance: jax.Array
    residual_variance: jax.Array
    prior_std: float

    @property
    def output_dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.mean.shape[1])

    def weight_variance(self) -> jax.Array:
        return jnp.exp(self.log_variance)


def initialize_layer(
    key: jax.Array,
    output_dim: int,
    input_dim: int,
    *,
    prior_std: float,
    residual_variance: float,
    initializer: str,
    initial_log_variance: float,
) -> Layer:
    """Initialize one Bayesian linear layer."""
    if initializer == "xavier":
        scale = jnp.sqrt(1.0 / input_dim)
    elif initializer == "he":
        scale = jnp.sqrt(2.0 / input_dim)
    else:
        raise ValueError(f"unsupported initializer: {initializer!r}")

    mean = jax.random.normal(key, (output_dim, input_dim)) * scale
    log_variance = clamp_weight_log_variance(
        jnp.full((output_dim, input_dim), float(initial_log_variance))
    )
    residual = jnp.full((output_dim,), float(residual_variance))
    return Layer(
        mean=mean,
        log_variance=log_variance,
        residual_variance=residual,
        prior_std=float(prior_std),
    )
