"""JIT-compiled BPCN minibatch update."""

from functools import partial
from typing import NamedTuple

import jax

from bpcn.configs.base import BPCNConfig
from bpcn.data.types import Batch
from bpcn.inference.e_step import EStepMetrics, infer_latents
from bpcn.inference.shared_energy import EnergyTerms, full_energy_terms
from bpcn.inference.state import LatentState
from bpcn.models.network import Network
from bpcn.training.gradients import LayerUpdateMetrics
from bpcn.training.m_step import MStepResult, m_step


class BatchStepResult(NamedTuple):
    network: Network
    latents: LatentState
    e_step_metrics: EStepMetrics
    layer_metrics: tuple[LayerUpdateMetrics, ...]
    energy_after: EnergyTerms


def _with_loop_summary(
    first: LayerUpdateMetrics, final: LayerUpdateMetrics
) -> LayerUpdateMetrics:
    return final._replace(
        loop_data_loss_before=first.data_loss_before,
        loop_data_loss_after=final.data_loss_after,
        loop_data_loss_delta=final.data_loss_after - first.data_loss_before,
    )


def make_batch_step(config: BPCNConfig, training_set_size: int):
    """Build a fused E-step and repeated-M-step minibatch function."""
    if training_set_size < 1:
        raise ValueError("training_set_size must be >= 1")
    update_steps = config.update.steps
    prior_scale = 1.0 / float(training_set_size)

    @partial(jax.jit, static_argnums=())
    def batch_step(network: Network, batch: Batch, key) -> BatchStepResult:
        data_scale = 1.0 / float(batch.inputs.shape[0])
        keys = jax.random.split(key, 2 + update_steps)
        inference_result = infer_latents(
            network, batch.inputs, batch.targets, config, keys[0]
        )

        updated_network = network
        first_metrics = None
        final_metrics = None
        for update_index in range(update_steps):
            result: MStepResult = m_step(
                updated_network,
                inference_result.latents,
                batch.inputs,
                batch.targets,
                config.update,
                data_scale=data_scale,
                prior_scale=prior_scale,
                key=keys[1 + update_index],
            )
            updated_network = result.network
            if first_metrics is None:
                first_metrics = result.layer_metrics
            final_metrics = result.layer_metrics

        layer_metrics = tuple(
            _with_loop_summary(first, final)
            for first, final in zip(first_metrics, final_metrics)
        )
        energy = full_energy_terms(
            updated_network,
            batch.inputs,
            batch.targets,
            inference_result.latents,
            config.update,
            key=keys[-1],
            weight_kl_scale=prior_scale,
        )
        return BatchStepResult(
            network=updated_network,
            latents=inference_result.latents,
            e_step_metrics=inference_result.metrics,
            layer_metrics=layer_metrics,
            energy_after=energy,
        )

    return batch_step
