"""BPCN gradient, update, and minibatch APIs."""

from bpcn.training.gradients import (
    LayerGradientTerms,
    LayerUpdateMetrics,
    LayerUpdateResult,
    apply_layer_update,
    compute_layer_gradients,
)
from bpcn.training.loop import BatchStepResult, make_batch_step
from bpcn.training.metrics import EpochMetricsAccumulator, EpochSummary
from bpcn.training.m_step import MStepResult, m_step

__all__ = [
    "BatchStepResult",
    "EpochMetricsAccumulator",
    "EpochSummary",
    "LayerGradientTerms",
    "LayerUpdateMetrics",
    "LayerUpdateResult",
    "MStepResult",
    "apply_layer_update",
    "compute_layer_gradients",
    "m_step",
    "make_batch_step",
]
