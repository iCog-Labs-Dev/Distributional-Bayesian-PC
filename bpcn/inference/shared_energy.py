"""Decomposed free energy shared by BPCN inference and training."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bpcn.configs.base import UpdateConfig
from bpcn.data.types import Targets
from bpcn.inference.state import LatentState
from bpcn.losses.categorical import categorical_output_loss
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.losses.weight_kl import gaussian_weight_kl_components
from bpcn.models.activations import activation_moments
from bpcn.models.moments import PredictiveMoments, moment_forward
from bpcn.models.network import Network


class EnergyTerms(NamedTuple):
    output: jax.Array
    transitions: jax.Array
    weight_mean_kl: jax.Array
    weight_variance_kl: jax.Array

    @property
    def weight_kl(self):
        return self.weight_mean_kl + self.weight_variance_kl

    @property
    def total(self):
        return self.output + self.transitions + self.weight_kl


def presynaptic_moments(
    network: Network, inputs, latents: LatentState, layer_index: int
):
    if layer_index == 0:
        return inputs, jnp.zeros_like(inputs)
    return activation_moments(
        network.hidden_activations[layer_index - 1],
        latents.means[layer_index - 1],
        latents.variances[layer_index - 1],
    )


def hidden_predictive_moments(
    network: Network, inputs, latents: LatentState, layer_index: int
) -> PredictiveMoments:
    input_mean, input_variance = presynaptic_moments(
        network, inputs, latents, layer_index
    )
    return moment_forward(
        network.layers[layer_index], input_mean, input_variance
    )


def output_predictive_moments(
    network: Network, latents: LatentState
) -> PredictiveMoments:
    feature_mean, feature_variance = activation_moments(
        network.hidden_activations[-1],
        latents.top_mean,
        latents.top_variance,
    )
    return moment_forward(network.output_layer, feature_mean, feature_variance)


def hidden_transition_energy(network: Network, inputs, latents: LatentState):
    total = jnp.zeros((), dtype=latents.means[0].dtype)
    for layer_index in range(network.hidden_layer_count):
        predictive = hidden_predictive_moments(
            network, inputs, latents, layer_index
        )
        terms = gaussian_kl(
            latents.means[layer_index],
            latents.variances[layer_index],
            predictive.mean,
            predictive.variance,
        )
        total = total + terms.value.sum(axis=-1).mean()
    return total


def output_energy(
    network: Network,
    latents: LatentState,
    targets: Targets,
    key,
    mc_samples: int,
):
    if network.output_likelihood == "gaussian":
        predictive = output_predictive_moments(network, latents)
        return gaussian_kl(
            targets.mean,
            targets.variance,
            predictive.mean,
            predictive.variance,
        ).value.sum(axis=-1).mean()
    if network.output_likelihood == "categorical":
        return categorical_output_loss(
            network.output_layer,
            network.hidden_activations[-1],
            network.output_estimator,
            latents.top_mean,
            latents.top_variance,
            targets.class_indices,
            key,
            mc_samples,
        )
    raise ValueError(f"unsupported output likelihood: {network.output_likelihood!r}")


def latent_free_energy(
    network: Network,
    inputs,
    latents: LatentState,
    *,
    targets: Targets | None,
    key,
    mc_samples: int,
    output_loss_weight: float,
):
    transition_term = hidden_transition_energy(network, inputs, latents)
    if targets is None:
        return transition_term
    return transition_term + output_loss_weight * output_energy(
        network, latents, targets, key, mc_samples
    )


def weight_kl_terms(
    network: Network, update: UpdateConfig, scale: float
):
    dtype = network.layers[0].mean.dtype
    mean_total = jnp.zeros((), dtype=dtype)
    variance_total = jnp.zeros((), dtype=dtype)

    hidden_mean_scale = (
        update.hidden_weight_kl_scale
        if update.hidden_mean_kl_scale is None
        else update.hidden_mean_kl_scale
    )
    output_mean_scale = (
        update.output_weight_kl_scale
        if update.output_mean_kl_scale is None
        else update.output_mean_kl_scale
    )
    for layer in network.layers[:-1]:
        components = gaussian_weight_kl_components(
            layer.mean, layer.log_variance, layer.prior_std
        )
        mean_total = mean_total + hidden_mean_scale * components.mean.sum()
        variance_total = (
            variance_total
            + update.hidden_weight_kl_scale * components.variance.sum()
        )
    output = network.output_layer
    components = gaussian_weight_kl_components(
        output.mean, output.log_variance, output.prior_std
    )
    mean_total = mean_total + output_mean_scale * components.mean.sum()
    variance_total = (
        variance_total
        + update.output_weight_kl_scale * components.variance.sum()
    )
    return scale * mean_total, scale * variance_total


def full_energy_terms(
    network: Network,
    inputs,
    targets: Targets,
    latents: LatentState,
    update: UpdateConfig,
    *,
    key,
    weight_kl_scale: float,
) -> EnergyTerms:
    output_term = update.output_loss_weight * output_energy(
        network,
        latents,
        targets,
        key,
        update.training_mc_samples,
    )
    transition_term = hidden_transition_energy(network, inputs, latents)
    mean_kl, variance_kl = weight_kl_terms(
        network, update, weight_kl_scale
    )
    return EnergyTerms(
        output=output_term,
        transitions=transition_term,
        weight_mean_kl=mean_kl,
        weight_variance_kl=variance_kl,
    )
