"""Numerical utilities for BPCN."""

from bpcn.utils.safe_math import (
    MAX_LATENT_LOG_VARIANCE,
    MAX_WEIGHT_LOG_VARIANCE,
    MIN_LATENT_LOG_VARIANCE,
    MIN_VARIANCE,
    MIN_WEIGHT_LOG_VARIANCE,
    clamp_latent_log_variance,
    clamp_weight_log_variance,
    floor_variance,
)

__all__ = [
    "MAX_LATENT_LOG_VARIANCE",
    "MAX_WEIGHT_LOG_VARIANCE",
    "MIN_LATENT_LOG_VARIANCE",
    "MIN_VARIANCE",
    "MIN_WEIGHT_LOG_VARIANCE",
    "clamp_latent_log_variance",
    "clamp_weight_log_variance",
    "floor_variance",
]
