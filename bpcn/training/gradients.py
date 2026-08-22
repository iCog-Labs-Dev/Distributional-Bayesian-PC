"""Analytical Gaussian-layer gradients and parameter updates."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.losses.weight_kl import gaussian_weight_kl
from bpcn.models.layer import Layer
from bpcn.models.moments import PredictiveMoments, moment_forward
from bpcn.utils.safe_math import (
    MAX_WEIGHT_LOG_VARIANCE,
    MIN_WEIGHT_LOG_VARIANCE,
    clamp_weight_log_variance,
)


class LayerGradientTerms(NamedTuple):
    predictive: PredictiveMoments
    input_second_moment: jax.Array
    mean_error: jax.Array
    variance_residual: jax.Array
    mean_data_gradient: jax.Array
    mean_prior_gradient: jax.Array
    mean_gradient: jax.Array
    log_variance_data_gradient: jax.Array
    log_variance_prior_gradient: jax.Array
    log_variance_gradient: jax.Array


class LayerUpdateMetrics(NamedTuple):
    data_loss_before: jax.Array
    data_loss_after: jax.Array
    data_loss_delta: jax.Array
    loop_data_loss_before: jax.Array
    loop_data_loss_after: jax.Array
    loop_data_loss_delta: jax.Array
    weight_kl: jax.Array
    mean_gradient_norm: jax.Array
    log_variance_gradient_norm: jax.Array
    mean_data_gradient_norm: jax.Array
    mean_prior_gradient_norm: jax.Array
    log_variance_data_gradient_norm: jax.Array
    log_variance_prior_gradient_norm: jax.Array
    mean_weight_variance: jax.Array
    positive_variance_residual_fraction: jax.Array
    mean_abs_residual: jax.Array
    log_variance_clamp_fraction: jax.Array
    finite: jax.Array


class LayerUpdateResult(NamedTuple):
    layer: Layer
    metrics: LayerUpdateMetrics


def compute_layer_gradients(
    layer: Layer,
    input_mean,
    input_variance,
    target_mean,
    target_variance,
    *,
    data_scale: float,
    prior_scale: float,
    weight_kl_scale: float,
    mean_kl_scale: float | None = None,
) -> LayerGradientTerms:
    """Compute the ascent direction for the negative Gaussian inclusion-KL."""
    predictive = moment_forward(layer, input_mean, input_variance)
    matching = gaussian_kl(
        target_mean,
        target_variance,
        predictive.mean,
        predictive.variance,
    )
    mean_error = matching.mean_error
    variance_residual = matching.variance_residual
    inverse_variance = 1.0 / matching.predictive_variance
    inverse_variance_squared = inverse_variance**2
    input_second_moment = input_mean**2 + input_variance
    weight_variance = layer.weight_variance()

    mean_data_gradient = data_scale * (
        jnp.einsum("bi,bj->ij", mean_error * inverse_variance, input_mean)
        + layer.mean
        * jnp.einsum(
            "bi,bj->ij",
            variance_residual * inverse_variance_squared,
            input_variance,
        )
    )
    log_variance_data_gradient = (
        data_scale
        * weight_variance
        * jnp.einsum(
            "bi,bj->ij",
            0.5 * variance_residual * inverse_variance_squared,
            input_second_moment,
        )
    )

    prior_variance = layer.prior_std**2
    effective_mean_scale = (
        weight_kl_scale if mean_kl_scale is None else mean_kl_scale
    )
    mean_prior_gradient = (
        -prior_scale * effective_mean_scale * layer.mean / prior_variance
    )
    log_variance_prior_gradient = (
        -prior_scale
        * 0.5
        * weight_kl_scale
        * (weight_variance / prior_variance - 1.0)
    )
    return LayerGradientTerms(
        predictive=predictive,
        input_second_moment=input_second_moment,
        mean_error=mean_error,
        variance_residual=variance_residual,
        mean_data_gradient=mean_data_gradient,
        mean_prior_gradient=mean_prior_gradient,
        mean_gradient=mean_data_gradient + mean_prior_gradient,
        log_variance_data_gradient=log_variance_data_gradient,
        log_variance_prior_gradient=log_variance_prior_gradient,
        log_variance_gradient=(
            log_variance_data_gradient + log_variance_prior_gradient
        ),
    )


def _mean_norm(value):
    return jnp.sqrt(jnp.sum(value**2))


def _predictive_data_loss(predictive, target_mean, target_variance):
    terms = gaussian_kl(
        target_mean,
        target_variance,
        predictive.mean,
        predictive.variance,
    )
    return terms.value.mean(), terms.variance_residual


def _data_loss(layer, input_mean, input_variance, target_mean, target_variance):
    predictive = moment_forward(layer, input_mean, input_variance)
    return _predictive_data_loss(predictive, target_mean, target_variance)


def apply_layer_update(
    layer: Layer,
    gradients: LayerGradientTerms,
    input_mean,
    input_variance,
    target_mean,
    target_variance,
    *,
    mean_learning_rate: float,
    log_variance_learning_rate: float,
) -> LayerUpdateResult:
    """Apply one ascent step and report concise numerical-health metrics."""
    loss_before, _ = _predictive_data_loss(
        gradients.predictive, target_mean, target_variance
    )
    updated_mean = layer.mean + mean_learning_rate * gradients.mean_gradient
    updated_log_variance = clamp_weight_log_variance(
        layer.log_variance
        + log_variance_learning_rate * gradients.log_variance_gradient
    )
    updated = Layer(
        mean=updated_mean,
        log_variance=updated_log_variance,
        residual_variance=layer.residual_variance,
        prior_std=layer.prior_std,
    )
    loss_after, residual_after = _data_loss(
        updated, input_mean, input_variance, target_mean, target_variance
    )
    weight_kl = gaussian_weight_kl(
        updated.mean, updated.log_variance, updated.prior_std
    ).mean()
    clamp_fraction = jnp.mean(
        (updated.log_variance <= MIN_WEIGHT_LOG_VARIANCE)
        | (updated.log_variance >= MAX_WEIGHT_LOG_VARIANCE)
    )
    finite = (
        jnp.all(jnp.isfinite(updated.mean))
        & jnp.all(jnp.isfinite(updated.log_variance))
        & jnp.isfinite(loss_after)
    )
    metrics = LayerUpdateMetrics(
        data_loss_before=loss_before,
        data_loss_after=loss_after,
        data_loss_delta=loss_after - loss_before,
        loop_data_loss_before=loss_before,
        loop_data_loss_after=loss_after,
        loop_data_loss_delta=loss_after - loss_before,
        weight_kl=weight_kl,
        mean_gradient_norm=_mean_norm(gradients.mean_gradient),
        log_variance_gradient_norm=_mean_norm(
            gradients.log_variance_gradient
        ),
        mean_data_gradient_norm=_mean_norm(gradients.mean_data_gradient),
        mean_prior_gradient_norm=_mean_norm(gradients.mean_prior_gradient),
        log_variance_data_gradient_norm=_mean_norm(
            gradients.log_variance_data_gradient
        ),
        log_variance_prior_gradient_norm=_mean_norm(
            gradients.log_variance_prior_gradient
        ),
        mean_weight_variance=updated.weight_variance().mean(),
        positive_variance_residual_fraction=(residual_after > 0).mean(),
        mean_abs_residual=jnp.abs(residual_after).mean(),
        log_variance_clamp_fraction=clamp_fraction,
        finite=finite,
    )
    return LayerUpdateResult(layer=updated, metrics=metrics)
