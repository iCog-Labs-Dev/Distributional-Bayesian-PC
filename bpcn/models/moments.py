"""Predictive moments for Bayesian linear layers."""

from typing import NamedTuple

import jax

from bpcn.models.layer import Layer


class PredictiveMoments(NamedTuple):
    mean: jax.Array
    variance: jax.Array


class VarianceComponents(NamedTuple):
    residual: jax.Array
    propagated: jax.Array
    epistemic: jax.Array


def variance_components(
    layer: Layer, input_mean: jax.Array, input_variance: jax.Array
) -> VarianceComponents:
    weight_variance = layer.weight_variance()
    propagated = input_variance @ (layer.mean**2).T
    epistemic = (input_mean**2 + input_variance) @ weight_variance.T
    return VarianceComponents(
        residual=layer.residual_variance,
        propagated=propagated,
        epistemic=epistemic,
    )


def moment_forward(
    layer: Layer, input_mean: jax.Array, input_variance: jax.Array
) -> PredictiveMoments:
    """Return the predictive mean and variance for one layer."""
    components = variance_components(layer, input_mean, input_variance)
    predictive_mean = input_mean @ layer.mean.T
    predictive_variance = (
        components.residual[None, :]
        + components.propagated
        + components.epistemic
    )
    return PredictiveMoments(predictive_mean, predictive_variance)
