"""Pure predictive metrics and variance summaries."""

from typing import NamedTuple

import jax.numpy as jnp

from bpcn.models.moments import variance_components


class VarianceDecomposition(NamedTuple):
    residual_total: float
    propagated_total: float
    epistemic_total: float
    total: float
    residual_fraction: float
    propagated_fraction: float
    epistemic_fraction: float


def predictive_entropy(probabilities):
    safe = jnp.clip(probabilities, 1e-12, 1.0)
    return -jnp.sum(safe * jnp.log(safe), axis=-1)


def summarize_variance(
    network, input_mean, input_variance, layer_index: int = -1
) -> VarianceDecomposition:
    """Summarize the three predictive-variance contributions for one layer."""
    components = variance_components(
        network.layers[layer_index], input_mean, input_variance
    )
    residual_total = components.residual.sum()
    propagated_total = components.propagated.mean(axis=0).sum()
    epistemic_total = components.epistemic.mean(axis=0).sum()
    total = residual_total + propagated_total + epistemic_total
    return VarianceDecomposition(
        residual_total=float(residual_total),
        propagated_total=float(propagated_total),
        epistemic_total=float(epistemic_total),
        total=float(total),
        residual_fraction=float(residual_total / total),
        propagated_fraction=float(propagated_total / total),
        epistemic_fraction=float(epistemic_total / total),
    )
