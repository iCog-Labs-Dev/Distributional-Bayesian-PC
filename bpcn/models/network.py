"""BPCN network construction and JAX pytree registration."""

import math
from typing import NamedTuple, Tuple

import jax

from bpcn.configs.base import ModelConfig
from bpcn.models.layer import Layer, initialize_layer


class Network(NamedTuple):
    """Hidden Bayesian layers followed by one Bayesian output layer."""

    layers: Tuple[Layer, ...]
    hidden_activations: Tuple[str, ...]
    output_likelihood: str
    output_estimator: str

    @property
    def hidden_layer_count(self) -> int:
        return len(self.layers) - 1

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]


def _flatten_network(network: Network):
    children = (network.layers,)
    metadata = (
        network.hidden_activations,
        network.output_likelihood,
        network.output_estimator,
    )
    return children, metadata


def _unflatten_network(metadata, children):
    (layers,) = children
    hidden_activations, output_likelihood, output_estimator = metadata
    return Network(
        layers=layers,
        hidden_activations=hidden_activations,
        output_likelihood=output_likelihood,
        output_estimator=output_estimator,
    )


jax.tree_util.register_pytree_node(Network, _flatten_network, _unflatten_network)


def initialize_network(key: jax.Array, config: ModelConfig) -> Network:
    """Initialize a network while preserving the configured prior scheme."""
    dimensions = config.layer_dims
    hidden_count = config.hidden_layer_count
    layer_count = hidden_count + 1

    if config.prior_scheme == "constant":
        prior_stds = [config.hidden_prior_std] * hidden_count + [
            config.output_prior_std
        ]
        initial_log_variances = [config.initial_weight_log_variance] * layer_count
    else:
        coefficient = 2.0 if config.prior_scheme == "matched_he" else 1.0
        prior_stds = [
            math.sqrt(coefficient / float(dimensions[index]))
            for index in range(layer_count)
        ]
        initial_log_variances = [
            math.log(coefficient / float(dimensions[index]))
            for index in range(layer_count)
        ]

    for index in range(hidden_count):
        initial_log_variances[index] += config.hidden_log_variance_offset

    keys = jax.random.split(key, layer_count)
    layers = []
    for index in range(hidden_count):
        layers.append(
            initialize_layer(
                keys[index],
                dimensions[index + 1],
                dimensions[index],
                prior_std=prior_stds[index],
                residual_variance=config.hidden_residual_variance,
                initializer=config.hidden_initializer,
                initial_log_variance=initial_log_variances[index],
            )
        )
    layers.append(
        initialize_layer(
            keys[-1],
            dimensions[-1],
            dimensions[-2],
            prior_std=prior_stds[-1],
            residual_variance=config.output_residual_variance,
            initializer=config.output_initializer,
            initial_log_variance=initial_log_variances[-1],
        )
    )
    return Network(
        layers=tuple(layers),
        hidden_activations=config.activations,
        output_likelihood=config.output_likelihood,
        output_estimator=config.output_estimator,
    )
