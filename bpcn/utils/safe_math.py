"""Numerical safeguards shared by BPCN computations."""

import jax.numpy as jnp


MIN_VARIANCE = 1e-8
MIN_WEIGHT_LOG_VARIANCE = -12.0
MAX_WEIGHT_LOG_VARIANCE = 6.0
MIN_LATENT_LOG_VARIANCE = -12.0
MAX_LATENT_LOG_VARIANCE = 6.0


def clamp_weight_log_variance(log_variance):
    return jnp.clip(
        log_variance, MIN_WEIGHT_LOG_VARIANCE, MAX_WEIGHT_LOG_VARIANCE
    )


def clamp_latent_log_variance(log_variance):
    return jnp.clip(
        log_variance, MIN_LATENT_LOG_VARIANCE, MAX_LATENT_LOG_VARIANCE
    )


def floor_variance(variance):
    return jnp.maximum(variance, MIN_VARIANCE)
