"""Target-free posterior-mean and Monte Carlo prediction."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from bpcn.configs.base import BPCNConfig, InferenceConfig
from bpcn.data.types import DatasetSplit
from bpcn.evaluation.metrics import predictive_entropy
from bpcn.inference.e_step import infer_target_free
from bpcn.models.activations import activation_moments, apply_activation_sample
from bpcn.models.network import Network


class PredictionResult(NamedTuple):
    mc_probabilities: np.ndarray
    mean_probabilities: np.ndarray


class EvaluationMetrics(NamedTuple):
    sample_count: int
    mc_accuracy: float
    mc_log_likelihood: float
    mc_entropy_mean: float
    mc_entropy_std: float
    mean_accuracy: float
    mean_log_likelihood: float
    mean_entropy_mean: float
    mean_entropy_std: float


def _mean_probabilities(network: Network, latents):
    feature_mean, _ = activation_moments(
        network.hidden_activations[-1],
        latents.top_mean,
        latents.top_variance,
    )
    return jax.nn.softmax(
        feature_mean @ network.output_layer.mean.T, axis=-1
    )


def _mc_probabilities(network: Network, latents, key, samples: int):
    output = network.output_layer
    weight_std = jnp.sqrt(output.weight_variance())
    latent_std = jnp.sqrt(latents.top_variance)

    def one_sample(sample_key):
        weight_key, latent_key = jax.random.split(sample_key)
        sampled_weight = output.mean + weight_std * jax.random.normal(
            weight_key, output.mean.shape
        )
        sampled_latent = (
            latents.top_mean
            + latent_std
            * jax.random.normal(latent_key, latents.top_mean.shape)
        )
        features = apply_activation_sample(
            network.hidden_activations[-1], sampled_latent
        )
        return jax.nn.softmax(features @ sampled_weight.T, axis=-1)

    keys = jax.random.split(key, samples)
    return jax.vmap(one_sample)(keys).mean(axis=0)


@partial(jax.jit, static_argnames=("inference", "mc_samples"))
def _predict_batch(
    network: Network,
    inputs,
    key,
    *,
    inference: InferenceConfig,
    mc_samples: int,
):
    latents = infer_target_free(network, inputs, inference).latents
    return (
        _mc_probabilities(network, latents, key, mc_samples),
        _mean_probabilities(network, latents),
    )


class _EvaluationScalars(NamedTuple):
    correct_mc: jax.Array
    correct_mean: jax.Array
    log_likelihood_mc: jax.Array
    log_likelihood_mean: jax.Array
    entropy_mc_mean: jax.Array
    entropy_mc_std: jax.Array
    entropy_mean_mean: jax.Array
    entropy_mean_std: jax.Array


@partial(
    jax.jit,
    static_argnames=(
        "batch_count",
        "batch_size",
        "inference",
        "mc_samples",
    ),
)
def _predict_padded(
    network,
    padded_inputs,
    keys,
    *,
    batch_count,
    batch_size,
    inference,
    mc_samples,
):
    batches = padded_inputs.reshape(batch_count, batch_size, -1)

    def step(_, arguments):
        key, inputs = arguments
        probabilities = _predict_batch(
            network,
            inputs,
            key,
            inference=inference,
            mc_samples=mc_samples,
        )
        return None, probabilities

    _, (mc_batches, mean_batches) = jax.lax.scan(
        step, None, (keys, batches)
    )
    return (
        mc_batches.reshape(batch_count * batch_size, -1),
        mean_batches.reshape(batch_count * batch_size, -1),
    )


@partial(
    jax.jit,
    static_argnames=(
        "batch_count",
        "batch_size",
        "inference",
        "mc_samples",
    ),
)
def _evaluate_padded(
    network,
    padded_inputs,
    padded_labels,
    mask,
    keys,
    *,
    batch_count,
    batch_size,
    inference,
    mc_samples,
) -> _EvaluationScalars:
    mc_probabilities, mean_probabilities = _predict_padded(
        network,
        padded_inputs,
        keys,
        batch_count=batch_count,
        batch_size=batch_size,
        inference=inference,
        mc_samples=mc_samples,
    )
    valid = mask.astype(mc_probabilities.dtype)
    correct_mc = jnp.sum(
        (jnp.argmax(mc_probabilities, axis=-1) == padded_labels) * valid
    )
    correct_mean = jnp.sum(
        (jnp.argmax(mean_probabilities, axis=-1) == padded_labels) * valid
    )
    row_indices = jnp.arange(mc_probabilities.shape[0])
    log_likelihood_mc = jnp.sum(
        jnp.log(
            jnp.clip(
                mc_probabilities[row_indices, padded_labels], 1e-12, 1.0
            )
        )
        * valid
    )
    log_likelihood_mean = jnp.sum(
        jnp.log(
            jnp.clip(
                mean_probabilities[row_indices, padded_labels], 1e-12, 1.0
            )
        )
        * valid
    )
    mc_entropy = predictive_entropy(mc_probabilities) * valid
    mean_entropy = predictive_entropy(mean_probabilities) * valid
    denominator = jnp.maximum(valid.sum(), 1.0)
    mc_entropy_mean = mc_entropy.sum() / denominator
    mean_entropy_mean = mean_entropy.sum() / denominator
    mc_entropy_std = jnp.sqrt(
        jnp.sum(((mc_entropy - mc_entropy_mean) * valid) ** 2)
        / denominator
    )
    mean_entropy_std = jnp.sqrt(
        jnp.sum(((mean_entropy - mean_entropy_mean) * valid) ** 2)
        / denominator
    )
    return _EvaluationScalars(
        correct_mc=correct_mc,
        correct_mean=correct_mean,
        log_likelihood_mc=log_likelihood_mc,
        log_likelihood_mean=log_likelihood_mean,
        entropy_mc_mean=mc_entropy_mean,
        entropy_mc_std=mc_entropy_std,
        entropy_mean_mean=mean_entropy_mean,
        entropy_mean_std=mean_entropy_std,
    )


def _padded_split(split: DatasetSplit, batch_size: int, key):
    sample_count = len(split.inputs)
    if sample_count < 1:
        raise ValueError("cannot evaluate an empty split")
    batch_count = (sample_count + batch_size - 1) // batch_size
    padded_count = batch_count * batch_size
    padding = padded_count - sample_count
    inputs = jnp.asarray(split.inputs)
    labels = jnp.asarray(split.class_indices)
    if padding:
        inputs = jnp.concatenate(
            [
                inputs,
                jnp.zeros((padding, inputs.shape[1]), dtype=inputs.dtype),
            ],
            axis=0,
        )
        labels = jnp.concatenate(
            [labels, jnp.zeros((padding,), dtype=labels.dtype)], axis=0
        )
    mask = jnp.arange(padded_count) < sample_count
    keys = jax.random.split(key, batch_count + 1)[1:]
    return sample_count, batch_count, inputs, labels, mask, keys


def predict_split(
    network: Network,
    split: DatasetSplit,
    config: BPCNConfig,
    key,
    *,
    batch_size: int | None = None,
) -> PredictionResult:
    evaluation = config.evaluation
    size = evaluation.batch_size if batch_size is None else int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be >= 1")
    sample_count, batch_count, inputs, _, _, keys = _padded_split(
        split, size, key
    )
    inference = evaluation.resolve_inference(config.inference)
    mc_probabilities, mean_probabilities = _predict_padded(
        network,
        inputs,
        keys,
        batch_count=batch_count,
        batch_size=size,
        inference=inference,
        mc_samples=evaluation.mc_samples,
    )
    return PredictionResult(
        mc_probabilities=np.asarray(mc_probabilities[:sample_count]),
        mean_probabilities=np.asarray(mean_probabilities[:sample_count]),
    )


def evaluate_split(
    network: Network,
    split: DatasetSplit,
    config: BPCNConfig,
    key,
    *,
    batch_size: int | None = None,
) -> EvaluationMetrics:
    evaluation = config.evaluation
    size = evaluation.batch_size if batch_size is None else int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be >= 1")
    sample_count, batch_count, inputs, labels, mask, keys = _padded_split(
        split, size, key
    )
    scalars = _evaluate_padded(
        network,
        inputs,
        labels,
        mask,
        keys,
        batch_count=batch_count,
        batch_size=size,
        inference=evaluation.resolve_inference(config.inference),
        mc_samples=evaluation.mc_samples,
    )
    count = float(sample_count)
    return EvaluationMetrics(
        sample_count=sample_count,
        mc_accuracy=float(scalars.correct_mc) / count,
        mc_log_likelihood=float(scalars.log_likelihood_mc) / count,
        mc_entropy_mean=float(scalars.entropy_mc_mean),
        mc_entropy_std=float(scalars.entropy_mc_std),
        mean_accuracy=float(scalars.correct_mean) / count,
        mean_log_likelihood=float(scalars.log_likelihood_mean) / count,
        mean_entropy_mean=float(scalars.entropy_mean_mean),
        mean_entropy_std=float(scalars.entropy_mean_std),
    )
