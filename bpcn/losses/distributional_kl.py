"""Gaussian inclusion-KL used for distributional matching."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bpcn.utils.safe_math import floor_variance


class GaussianKLTerms(NamedTuple):
    value: jax.Array
    mean_error: jax.Array
    variance_residual: jax.Array
    predictive_variance: jax.Array


def gaussian_kl(
    target_mean,
    target_variance,
    predictive_mean,
    predictive_variance,
) -> GaussianKLTerms:
    """Return the per-unit KL and its mean/variance residuals."""
    predictive_safe = floor_variance(predictive_variance)
    target_safe = floor_variance(target_variance)
    mean_error = target_mean - predictive_mean
    variance_residual = (
        target_variance + mean_error**2 - predictive_variance
    )
    value = 0.5 * (
        jnp.log(predictive_safe)
        - jnp.log(target_safe)
        + (target_variance + mean_error**2) / predictive_safe
        - 1.0
    )
    return GaussianKLTerms(
        value=value,
        mean_error=mean_error,
        variance_residual=variance_residual,
        predictive_variance=predictive_safe,
    )
