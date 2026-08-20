"""Latent E-step for distributional predictive coding."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bpcn.configs.base import BPCNConfig, InferenceConfig
from bpcn.data.types import Targets
from bpcn.inference.shared_energy import latent_free_energy
from bpcn.inference.state import LatentState
from bpcn.models.activations import activation_moments
from bpcn.models.moments import moment_forward
from bpcn.models.network import Network
from bpcn.utils.safe_math import clamp_latent_log_variance


class EStepMetrics(NamedTuple):
    initial_energy: jax.Array
    final_energy: jax.Array
    energy_trace: jax.Array


class EStepResult(NamedTuple):
    latents: LatentState
    metrics: EStepMetrics


def _initialize_latents(
    network: Network,
    inputs,
    config: InferenceConfig,
    perturbation_key,
):
    variance_floor = jnp.asarray(config.initial_variance, dtype=inputs.dtype)
    feature_mean, feature_variance = inputs, jnp.zeros_like(inputs)
    means = []
    log_variances = []
    for layer_index in range(network.hidden_layer_count):
        predictive = moment_forward(
            network.layers[layer_index], feature_mean, feature_variance
        )
        mean = predictive.mean
        if config.initial_mean_perturbation_std > 0.0:
            layer_key = jax.random.fold_in(perturbation_key, layer_index)
            noise = jax.random.normal(layer_key, mean.shape, dtype=mean.dtype)
            mean = mean + (
                config.initial_mean_perturbation_std
                * jnp.sqrt(jnp.maximum(predictive.variance, variance_floor))
                * noise
            )
        log_variance = jnp.log(
            jnp.maximum(predictive.variance, variance_floor)
        )
        means.append(mean)
        log_variances.append(log_variance)
        if layer_index + 1 < network.hidden_layer_count:
            feature_mean, feature_variance = activation_moments(
                network.hidden_activations[layer_index],
                mean,
                predictive.variance,
            )
    return tuple(means), tuple(log_variances)


def _run_e_step(
    network: Network,
    inputs,
    targets: Targets | None,
    config: InferenceConfig,
    *,
    key,
    output_loss_weight: float,
    mc_samples: int,
) -> EStepResult:
    frozen_network = jax.lax.stop_gradient(network)
    if key is None:
        key = jax.random.PRNGKey(0)

    if config.initial_mean_perturbation_std > 0.0:
        perturbation_key, key = jax.random.split(key)
    else:
        perturbation_key = key
    initial_means, initial_log_variances = _initialize_latents(
        frozen_network, inputs, config, perturbation_key
    )

    def objective(means, log_variances, step_key):
        state = LatentState(
            means=means,
            variances=jax.tree.map(jnp.exp, log_variances),
        )
        return latent_free_energy(
            frozen_network,
            inputs,
            state,
            targets=targets,
            key=step_key,
            mc_samples=mc_samples,
            output_loss_weight=output_loss_weight,
        )

    gradient = jax.grad(objective, argnums=(0, 1))

    def step(carry, step_key):
        means, log_variances = carry
        mean_gradients, variance_gradients = gradient(
            means, log_variances, step_key
        )
        updated_means = jax.tree.map(
            lambda value, grad: value - config.mean_learning_rate * grad,
            means,
            mean_gradients,
        )
        updated_log_variances = jax.tree.map(
            lambda value, grad: clamp_latent_log_variance(
                value - config.log_variance_learning_rate * grad
            ),
            log_variances,
            variance_gradients,
        )
        energy = objective(updated_means, updated_log_variances, step_key)
        return (updated_means, updated_log_variances), energy

    step_keys = jax.random.split(key, config.steps)
    (final_means, final_log_variances), energy_trace = jax.lax.scan(
        step,
        (initial_means, initial_log_variances),
        step_keys,
        length=config.steps,
    )
    initial_energy = objective(
        initial_means, initial_log_variances, step_keys[0]
    )
    latents = LatentState(
        means=tuple(jax.lax.stop_gradient(value) for value in final_means),
        variances=tuple(
            jax.lax.stop_gradient(jnp.exp(value))
            for value in final_log_variances
        ),
    )
    return EStepResult(
        latents=latents,
        metrics=EStepMetrics(
            initial_energy=initial_energy,
            final_energy=energy_trace[-1],
            energy_trace=energy_trace,
        ),
    )


def infer_latents(
    network: Network,
    inputs,
    targets: Targets,
    config: BPCNConfig,
    key,
) -> EStepResult:
    """Infer supervised latent moments while holding network weights fixed."""
    if network.output_likelihood == "categorical" and targets.class_indices is None:
        raise ValueError("categorical inference requires class indices")
    return _run_e_step(
        network,
        inputs,
        targets,
        config.inference,
        key=key,
        output_loss_weight=config.update.output_loss_weight,
        mc_samples=config.update.training_mc_samples,
    )


def infer_target_free(
    network: Network,
    inputs,
    config: InferenceConfig,
    key=None,
) -> EStepResult:
    """Infer latent moments using hidden transition energy only."""
    return _run_e_step(
        network,
        inputs,
        None,
        config,
        key=key,
        output_loss_weight=0.0,
        mc_samples=1,
    )
