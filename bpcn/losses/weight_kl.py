"""KL between a diagonal-Gaussian weight posterior and Gaussian prior."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class WeightKLComponents(NamedTuple):
    mean: jax.Array
    variance: jax.Array

    @property
    def total(self):
        return self.mean + self.variance


def gaussian_weight_kl_components(
    mean, log_variance, prior_std, prior_mean=0.0
) -> WeightKLComponents:
    weight_variance = jnp.exp(log_variance)
    prior_variance = prior_std * prior_std
    mean_term = 0.5 * (mean - prior_mean) ** 2 / prior_variance
    variance_term = 0.5 * (
        weight_variance / prior_variance
        - 1.0
        + jnp.log(prior_variance)
        - log_variance
    )
    return WeightKLComponents(mean=mean_term, variance=variance_term)


def gaussian_weight_kl(mean, log_variance, prior_std, prior_mean=0.0):
    return gaussian_weight_kl_components(
        mean, log_variance, prior_std, prior_mean
    ).total
