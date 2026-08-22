"""Categorical output losses for posterior-mean and Monte Carlo estimators."""

import jax
import jax.numpy as jnp

from bpcn.models.activations import (
    activation_moments,
    apply_activation_sample,
)
from bpcn.models.layer import Layer


def per_example_nll(logits, class_indices):
    log_probabilities = jax.nn.log_softmax(logits, axis=-1)
    return -log_probabilities[jnp.arange(class_indices.shape[0]), class_indices]


def mean_categorical_loss(
    output: Layer,
    activation: str,
    latent_mean,
    latent_variance,
    class_indices,
):
    feature_mean, _ = activation_moments(
        activation, latent_mean, latent_variance
    )
    return per_example_nll(feature_mean @ output.mean.T, class_indices).mean()


def mc_categorical_loss(
    output: Layer,
    activation: str,
    latent_mean,
    latent_variance,
    class_indices,
    key,
    samples: int,
):
    weight_std = jnp.sqrt(output.weight_variance())
    latent_std = jnp.sqrt(latent_variance)

    def one_sample(sample_key):
        weight_key, latent_key = jax.random.split(sample_key)
        sampled_weight = output.mean + weight_std * jax.random.normal(
            weight_key, output.mean.shape
        )
        sampled_latent = latent_mean + latent_std * jax.random.normal(
            latent_key, latent_mean.shape
        )
        features = apply_activation_sample(activation, sampled_latent)
        return per_example_nll(features @ sampled_weight.T, class_indices)

    keys = jax.random.split(key, samples)
    return jax.vmap(one_sample)(keys).mean()


def categorical_output_loss(
    output: Layer,
    activation: str,
    estimator: str,
    latent_mean,
    latent_variance,
    class_indices,
    key,
    samples: int,
):
    if estimator == "mean":
        return mean_categorical_loss(
            output, activation, latent_mean, latent_variance, class_indices
        )
    if estimator == "mc":
        return mc_categorical_loss(
            output,
            activation,
            latent_mean,
            latent_variance,
            class_indices,
            key,
            samples,
        )
    raise ValueError(f"unsupported output estimator: {estimator!r}")
