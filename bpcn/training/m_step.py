"""Weight M-step for Gaussian hidden layers and Gaussian/categorical outputs."""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from bpcn.configs.base import UpdateConfig
from bpcn.data.types import Targets
from bpcn.inference.shared_energy import presynaptic_moments
from bpcn.inference.state import LatentState
from bpcn.losses.categorical import categorical_output_loss
from bpcn.losses.weight_kl import (
    gaussian_weight_kl,
    gaussian_weight_kl_components,
)
from bpcn.models.activations import activation_moments
from bpcn.models.layer import Layer
from bpcn.models.network import Network
from bpcn.training.gradients import (
    LayerUpdateMetrics,
    apply_layer_update,
    compute_layer_gradients,
)
from bpcn.utils.safe_math import (
    MAX_WEIGHT_LOG_VARIANCE,
    MIN_WEIGHT_LOG_VARIANCE,
    clamp_weight_log_variance,
)


class MStepResult(NamedTuple):
    network: Network
    layer_metrics: Tuple[LayerUpdateMetrics, ...]


def _norm(value):
    return jnp.sqrt(jnp.sum(value**2))


def _categorical_objective(
    mean,
    log_variance,
    output: Layer,
    network: Network,
    latents: LatentState,
    class_indices,
    key,
    update: UpdateConfig,
    data_scale: float,
    prior_scale: float,
):
    candidate = Layer(
        mean=mean,
        log_variance=log_variance,
        residual_variance=output.residual_variance,
        prior_std=output.prior_std,
    )
    loss = categorical_output_loss(
        candidate,
        network.hidden_activations[-1],
        network.output_estimator,
        latents.top_mean,
        latents.top_variance,
        class_indices,
        key,
        update.training_mc_samples,
    )
    batch_size = class_indices.shape[0]
    data_term = update.output_loss_weight * data_scale * batch_size * loss
    components = gaussian_weight_kl_components(
        mean, log_variance, output.prior_std
    )
    mean_scale = (
        update.output_weight_kl_scale
        if update.output_mean_kl_scale is None
        else update.output_mean_kl_scale
    )
    prior_term = prior_scale * (
        mean_scale * components.mean.sum()
        + update.output_weight_kl_scale * components.variance.sum()
    )
    return data_term + prior_term, loss


def _update_categorical_output(
    output: Layer,
    network: Network,
    latents: LatentState,
    class_indices,
    key,
    update: UpdateConfig,
    data_scale: float,
    prior_scale: float,
):
    def objective(mean, log_variance):
        return _categorical_objective(
            mean,
            log_variance,
            output,
            network,
            latents,
            class_indices,
            key,
            update,
            data_scale,
            prior_scale,
        )

    (_, loss_before), (mean_gradient, log_variance_gradient) = (
        jax.value_and_grad(objective, argnums=(0, 1), has_aux=True)(
            output.mean, output.log_variance
        )
    )
    prior_variance = output.prior_std**2
    mean_scale = (
        update.output_weight_kl_scale
        if update.output_mean_kl_scale is None
        else update.output_mean_kl_scale
    )
    mean_prior_gradient = (
        prior_scale * mean_scale * output.mean / prior_variance
    )
    log_variance_prior_gradient = (
        prior_scale
        * 0.5
        * update.output_weight_kl_scale
        * (output.weight_variance() / prior_variance - 1.0)
    )
    mean_data_gradient = mean_gradient - mean_prior_gradient
    log_variance_data_gradient = (
        log_variance_gradient - log_variance_prior_gradient
    )

    updated = Layer(
        mean=output.mean
        - update.output_mean_learning_rate * mean_gradient,
        log_variance=clamp_weight_log_variance(
            output.log_variance
            - update.output_log_variance_learning_rate
            * log_variance_gradient
        ),
        residual_variance=output.residual_variance,
        prior_std=output.prior_std,
    )
    _, loss_after = objective(updated.mean, updated.log_variance)
    clamp_fraction = jnp.mean(
        (updated.log_variance <= MIN_WEIGHT_LOG_VARIANCE)
        | (updated.log_variance >= MAX_WEIGHT_LOG_VARIANCE)
    )
    metrics = LayerUpdateMetrics(
        data_loss_before=loss_before,
        data_loss_after=loss_after,
        data_loss_delta=loss_after - loss_before,
        loop_data_loss_before=loss_before,
        loop_data_loss_after=loss_after,
        loop_data_loss_delta=loss_after - loss_before,
        weight_kl=gaussian_weight_kl(
            updated.mean, updated.log_variance, updated.prior_std
        ).mean(),
        mean_gradient_norm=_norm(mean_gradient),
        log_variance_gradient_norm=_norm(log_variance_gradient),
        mean_data_gradient_norm=_norm(mean_data_gradient),
        mean_prior_gradient_norm=_norm(mean_prior_gradient),
        log_variance_data_gradient_norm=_norm(
            log_variance_data_gradient
        ),
        log_variance_prior_gradient_norm=_norm(
            log_variance_prior_gradient
        ),
        mean_weight_variance=updated.weight_variance().mean(),
        positive_variance_residual_fraction=jnp.zeros(
            (), dtype=updated.mean.dtype
        ),
        mean_abs_residual=loss_after,
        log_variance_clamp_fraction=clamp_fraction,
        finite=(
            jnp.all(jnp.isfinite(updated.mean))
            & jnp.all(jnp.isfinite(updated.log_variance))
            & jnp.isfinite(loss_after)
        ),
    )
    return updated, metrics


def m_step(
    network: Network,
    latents: LatentState,
    inputs,
    targets: Targets,
    update: UpdateConfig,
    *,
    data_scale: float,
    prior_scale: float,
    key,
) -> MStepResult:
    """Apply one complete layerwise M-step."""
    new_layers = []
    metrics = []
    for layer_index in range(network.hidden_layer_count):
        input_mean, input_variance = presynaptic_moments(
            network, inputs, latents, layer_index
        )
        gradients = compute_layer_gradients(
            network.layers[layer_index],
            input_mean,
            input_variance,
            latents.means[layer_index],
            latents.variances[layer_index],
            data_scale=data_scale,
            prior_scale=prior_scale,
            weight_kl_scale=update.hidden_weight_kl_scale,
            mean_kl_scale=update.hidden_mean_kl_scale,
        )
        result = apply_layer_update(
            network.layers[layer_index],
            gradients,
            input_mean,
            input_variance,
            latents.means[layer_index],
            latents.variances[layer_index],
            mean_learning_rate=update.hidden_mean_learning_rate,
            log_variance_learning_rate=(
                update.hidden_log_variance_learning_rate
            ),
        )
        new_layers.append(result.layer)
        metrics.append(result.metrics)

    output = network.output_layer
    if network.output_likelihood == "gaussian":
        input_mean, input_variance = activation_moments(
            network.hidden_activations[-1],
            latents.top_mean,
            latents.top_variance,
        )
        gradients = compute_layer_gradients(
            output,
            input_mean,
            input_variance,
            targets.mean,
            targets.variance,
            data_scale=data_scale,
            prior_scale=prior_scale,
            weight_kl_scale=update.output_weight_kl_scale,
            mean_kl_scale=update.output_mean_kl_scale,
        )
        result = apply_layer_update(
            output,
            gradients,
            input_mean,
            input_variance,
            targets.mean,
            targets.variance,
            mean_learning_rate=update.output_mean_learning_rate,
            log_variance_learning_rate=(
                update.output_log_variance_learning_rate
            ),
        )
        new_output, output_metrics = result.layer, result.metrics
    elif network.output_likelihood == "categorical":
        if targets.class_indices is None:
            raise ValueError("categorical M-step requires class indices")
        new_output, output_metrics = _update_categorical_output(
            output,
            network,
            latents,
            jax.lax.stop_gradient(targets.class_indices),
            key,
            update,
            data_scale,
            prior_scale,
        )
    else:
        raise ValueError(
            f"unsupported output likelihood: {network.output_likelihood!r}"
        )
    new_layers.append(new_output)
    metrics.append(output_metrics)
    return MStepResult(
        network=network._replace(layers=tuple(new_layers)),
        layer_metrics=tuple(metrics),
    )
