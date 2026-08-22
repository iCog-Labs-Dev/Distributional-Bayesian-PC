"""One-shot linear-Gaussian calibration experiment for BPCN.

The scientific protocol is fixed below. Edit ``OUTPUT_DIR`` if needed, then
run ``python -m experiments.linear_gaussian``. The runner writes one concise
``results.json`` artifact and intentionally has no command-line flags.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from bpcn import (
    BPCNConfig,
    Batch,
    InferenceConfig,
    ModelConfig,
    Network,
    UpdateConfig,
    compute_layer_gradients,
    infer_latents,
    initialize_network,
    make_batch_step,
)
from bpcn.models import activation_moments, moment_forward
from bpcn.utils import MIN_WEIGHT_LOG_VARIANCE
from experiments._artifacts import atomic_write_json


_LOG = logging.getLogger(__name__)
_SCHEMA_VERSION = 1
_STAGES = ("A_output", "B_hidden")
_CASES = ("independent", "duplicate_agree", "collinear")
_TARGET_VARIANCE_MODES = ("matched", "production")


# The only routinely edited setting for this fixed one-shot experiment.
OUTPUT_DIR = Path("runs/linear_gaussian")


@dataclass(frozen=True)
class _Protocol:
    """Fixed scientific and numerical settings, grouped for validation/tests."""

    dimension: int = 8
    training_samples: int = 256
    test_samples: int = 512
    batch_size: int = 64
    collinear_rank: int = 4

    prior_std: float = 1.0
    output_residual_variances: tuple[float, ...] = (1e-2, 1e-1)
    stage_a_hidden_residual_variance: float = 1.0
    stage_b_hidden_residual_variance: float = 1e-2
    production_target_variance: float = 1e-3

    inference_steps: int = 20
    inference_mean_learning_rate: float = 0.1
    inference_log_variance_learning_rate: float = 0.05
    initial_latent_variance: float = 1e-8

    update_steps: int = 16
    weight_mean_learning_rate: float = 5e-2
    weight_log_variance_learning_rate: float = 0.5

    max_epochs: int = 2_000
    convergence_tolerance: float = 1e-6
    convergence_patience: int = 5
    numerical_floor_tolerance: float = 1e-5
    plateau_relative_change: float = 0.2

    data_seed: int = 20_260_806
    network_seed: int = 7
    training_seed: int = 11

    def __post_init__(self) -> None:
        integer_fields = (
            "dimension",
            "training_samples",
            "test_samples",
            "batch_size",
            "collinear_rank",
            "inference_steps",
            "update_steps",
            "max_epochs",
            "convergence_patience",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.collinear_rank >= self.dimension:
            raise ValueError("collinear_rank must be smaller than dimension")
        if self.training_samples % self.batch_size:
            raise ValueError("training_samples must be divisible by batch_size")
        if self.training_samples % 2:
            raise ValueError("training_samples must be even for paired inputs")

        positive_fields = (
            "prior_std",
            "stage_a_hidden_residual_variance",
            "stage_b_hidden_residual_variance",
            "production_target_variance",
            "inference_mean_learning_rate",
            "inference_log_variance_learning_rate",
            "initial_latent_variance",
            "weight_mean_learning_rate",
            "weight_log_variance_learning_rate",
            "convergence_tolerance",
            "numerical_floor_tolerance",
            "plateau_relative_change",
        )
        for name in positive_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        levels = tuple(float(value) for value in self.output_residual_variances)
        if not levels or any(not math.isfinite(value) or value <= 0.0 for value in levels):
            raise ValueError("output_residual_variances must contain positive values")
        object.__setattr__(self, "output_residual_variances", levels)


_PROTOCOL = _Protocol()


GATE_DEFINITIONS: Mapping[str, Mapping[str, Any]] = {
    "G_CAL_MEAN": {
        "metric": "relative_mean_error",
        "population": "independent inputs",
        "direction": "<=",
        "threshold": 0.05,
    },
    "G_CAL_VAR": {
        "metric": "relative_variance_error",
        "population": "independent inputs",
        "direction": "<=",
        "threshold": 0.10,
    },
    "G_CAL_COVERAGE": {
        "metric": "coverage_95",
        "population": "independent inputs",
        "direction": "within",
        "threshold": (0.90, 0.99),
    },
    "G_PREC_DIRECTION": {
        "metric": "precision_gain_ratio_dbpcn_over_exact",
        "population": "duplicate_agree versus independent",
        "direction": "within",
        "threshold": (0.5, 2.0),
    },
    "G_PRIOR_RETENTION": {
        "metric": "relative_prior_retention_error",
        "population": "data-free input coordinates",
        "direction": "<=",
        "threshold": 0.10,
    },
}


@dataclass(frozen=True)
class CellSpec:
    stage: str
    case: str
    output_residual_variance: float
    target_variance_mode: str

    @property
    def cell_id(self) -> str:
        noise = f"{self.output_residual_variance:g}".replace(".", "p").replace(
            "-", "m"
        )
        return (
            f"{self.stage}__{self.case}__b{noise}__"
            f"yv_{self.target_variance_mode}"
        )


@dataclass(frozen=True)
class _SyntheticData:
    train_inputs: np.ndarray
    train_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    true_weight: np.ndarray


@dataclass(frozen=True)
class _ExactPosterior:
    precision: np.ndarray
    covariance: np.ndarray
    row_blocks: np.ndarray
    mean: np.ndarray
    diagonal_covariance: np.ndarray
    inverse_diagonal_precision: np.ndarray
    gram: np.ndarray
    noise_covariance: np.ndarray


@dataclass(frozen=True)
class _TrainingOutcome:
    network: Network
    status: str
    epochs: int
    converged: bool
    final_mean_delta: float | None
    final_log_variance_delta: float | None
    history_tail: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class CellMetrics:
    relative_mean_error: float | None
    relative_variance_error: float | None
    relative_kl_optimal_variance_error: float | None
    coverage_95: float
    mean_weight_variance: float
    mean_exact_diagonal_covariance: float
    weight_variance_ratio: float | None
    total_precision_dbpcn: float
    total_precision_exact: float
    dbpcn_predictive_nll: float
    exact_predictive_nll: float
    relative_prior_retention_error: float | None
    mean_unconstrained_weight_variance: float | None
    mean_unconstrained_exact_variance: float | None


@dataclass(frozen=True)
class StationarityMetrics:
    batch_count: int
    within_batch_mean_drift: float | None
    within_batch_log_variance_drift: float
    residual_interpretable: bool | None
    relative_mean_gradient_residual: float | None
    relative_log_variance_gradient_residual: float | None
    mean_gradient_residual_norm: float
    log_variance_gradient_residual_norm: float
    mean_variance_residual: float
    mean_absolute_variance_residual: float
    exposure_weighted_variance_residual: float | None
    relative_exposure_weighted_variance_residual: float | None
    predictive_variance_relation_error: float | None


@dataclass(frozen=True)
class CellResult:
    spec: CellSpec
    status: str
    converged: bool
    epochs: int
    final_mean_delta: float | None
    final_log_variance_delta: float | None
    target_variance: float
    hidden_residual_variance: float
    noise_diagonal_mean: float
    noise_off_diagonal_max: float
    metrics: CellMetrics | None
    stationarity: StationarityMetrics | None
    history_tail: tuple[Mapping[str, float], ...]


@dataclass(frozen=True)
class LinearGaussianRunResult:
    protocol: Mapping[str, Any]
    sanity: Mapping[str, Any]
    cells: tuple[CellResult, ...]
    gate_definitions: Mapping[str, Mapping[str, Any]]
    gates: Mapping[str, Any]
    classification: Mapping[str, Any]


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if denominator == 0.0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _relative_error(observed: np.ndarray, reference: np.ndarray) -> float | None:
    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return _safe_ratio(
        float(np.linalg.norm(observed - reference)),
        float(np.linalg.norm(reference)),
    )


def _cell_grid(protocol: _Protocol = _PROTOCOL) -> tuple[CellSpec, ...]:
    return tuple(
        CellSpec(stage, case, output_variance, target_mode)
        for stage in _STAGES
        for case in _CASES
        for output_variance in protocol.output_residual_variances
        for target_mode in _TARGET_VARIANCE_MODES
    )


def _make_inputs(
    case: str, sample_count: int, rng: np.random.Generator, protocol: _Protocol
) -> np.ndarray:
    if case == "independent":
        return rng.standard_normal((sample_count, protocol.dimension))
    if case == "duplicate_agree":
        if sample_count % 2:
            raise ValueError("duplicate_agree requires an even sample count")
        base = rng.standard_normal((sample_count // 2, protocol.dimension))
        return np.repeat(base, 2, axis=0)
    if case == "collinear":
        inputs = rng.standard_normal((sample_count, protocol.dimension))
        inputs[:, protocol.collinear_rank :] = 0.0
        return inputs
    raise ValueError(f"unsupported linear-Gaussian case: {case!r}")


def _hidden_residual_variance(stage: str, protocol: _Protocol) -> float:
    if stage == "A_output":
        return protocol.stage_a_hidden_residual_variance
    if stage == "B_hidden":
        return protocol.stage_b_hidden_residual_variance
    raise ValueError(f"unsupported linear-Gaussian stage: {stage!r}")


def _make_dataset(
    spec: CellSpec, protocol: _Protocol = _PROTOCOL
) -> _SyntheticData:
    """Sample the two-stage model z=W1x+eps1, y=W2z+eps2."""
    rng = np.random.default_rng(protocol.data_seed)
    dimension = protocol.dimension
    true_weight = rng.standard_normal((dimension, dimension)) * protocol.prior_std
    hidden_variance = _hidden_residual_variance(spec.stage, protocol)

    def emit(inputs: np.ndarray) -> np.ndarray:
        sample_count = len(inputs)
        hidden_noise = math.sqrt(hidden_variance) * rng.standard_normal(
            (sample_count, dimension)
        )
        output_noise = math.sqrt(spec.output_residual_variance) * rng.standard_normal(
            (sample_count, dimension)
        )
        if spec.stage == "A_output":
            hidden = inputs + hidden_noise
            return hidden @ true_weight.T + output_noise
        hidden = inputs @ true_weight.T + hidden_noise
        return hidden + output_noise

    train_inputs = _make_inputs(
        spec.case, protocol.training_samples, rng, protocol
    )
    train_targets = emit(train_inputs)
    test_inputs = rng.standard_normal((protocol.test_samples, dimension))
    if spec.case == "collinear":
        test_inputs[:, protocol.collinear_rank :] = 0.0
    test_targets = emit(test_inputs)
    return _SyntheticData(
        train_inputs=train_inputs.astype(np.float64),
        train_targets=train_targets.astype(np.float64),
        test_inputs=test_inputs.astype(np.float64),
        test_targets=test_targets.astype(np.float64),
        true_weight=true_weight.astype(np.float64),
    )


def _noise_covariance(
    spec: CellSpec, true_weight: np.ndarray, protocol: _Protocol = _PROTOCOL
) -> np.ndarray:
    hidden_variance = _hidden_residual_variance(spec.stage, protocol)
    dimension = protocol.dimension
    if spec.stage == "A_output":
        output_map = np.asarray(true_weight, dtype=np.float64)
    else:
        output_map = np.eye(dimension, dtype=np.float64)
    return (
        spec.output_residual_variance * np.eye(dimension)
        + hidden_variance * (output_map @ output_map.T)
    )


def _exact_posterior(
    inputs: np.ndarray,
    targets: np.ndarray,
    noise_covariance: np.ndarray,
    prior_std: float,
) -> _ExactPosterior:
    """Closed-form posterior for y=Wx+eps with a spherical weight prior."""
    inputs = np.asarray(inputs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    noise_covariance = np.asarray(noise_covariance, dtype=np.float64)
    if inputs.ndim != 2 or targets.ndim != 2:
        raise ValueError("inputs and targets must be two-dimensional")
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have equal sample counts")
    output_dim, input_dim = targets.shape[1], inputs.shape[1]
    if noise_covariance.shape != (output_dim, output_dim):
        raise ValueError("noise covariance has an incompatible shape")

    gram = inputs.T @ inputs
    noise_precision = np.linalg.inv(noise_covariance)
    weight_count = output_dim * input_dim
    precision = (
        np.eye(weight_count) / (prior_std**2)
        + np.kron(noise_precision, gram)
    )
    covariance = np.linalg.inv(precision)
    right_hand_side = (noise_precision @ (targets.T @ inputs)).reshape(-1)
    mean = (covariance @ right_hand_side).reshape(output_dim, input_dim)
    diagonal_covariance = np.diag(covariance).reshape(output_dim, input_dim)
    inverse_diagonal_precision = (1.0 / np.diag(precision)).reshape(
        output_dim, input_dim
    )
    row_blocks = np.stack(
        [
            covariance[
                row * input_dim : (row + 1) * input_dim,
                row * input_dim : (row + 1) * input_dim,
            ]
            for row in range(output_dim)
        ]
    )
    return _ExactPosterior(
        precision=precision,
        covariance=covariance,
        row_blocks=row_blocks,
        mean=mean,
        diagonal_covariance=diagonal_covariance,
        inverse_diagonal_precision=inverse_diagonal_precision,
        gram=gram,
        noise_covariance=noise_covariance,
    )


def _exact_predictive_nll(
    posterior: _ExactPosterior,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
) -> float:
    mean = test_inputs @ posterior.mean.T
    parameter_variance = np.einsum(
        "nd,cde,ne->nc", test_inputs, posterior.row_blocks, test_inputs
    )
    variance = np.diag(posterior.noise_covariance)[None, :] + parameter_variance
    residual = test_targets - mean
    per_dimension = 0.5 * (
        np.log(2.0 * np.pi * variance) + residual**2 / variance
    )
    return float(per_dimension.sum(axis=1).mean())


def _resolve_target_variance(spec: CellSpec, protocol: _Protocol) -> float:
    if spec.target_variance_mode == "matched":
        return float(spec.output_residual_variance)
    if spec.target_variance_mode == "production":
        return float(protocol.production_target_variance)
    raise ValueError(f"unsupported target variance mode: {spec.target_variance_mode!r}")


def _build_bpcn_config(
    spec: CellSpec, protocol: _Protocol = _PROTOCOL
) -> BPCNConfig:
    learn_hidden = spec.stage == "B_hidden"
    model = ModelConfig(
        input_dim=protocol.dimension,
        hidden_dims=(protocol.dimension,),
        output_dim=protocol.dimension,
        activations=("identity",),
        initial_weight_log_variance=math.log(protocol.prior_std**2),
        hidden_prior_std=protocol.prior_std,
        output_prior_std=protocol.prior_std,
        hidden_residual_variance=_hidden_residual_variance(spec.stage, protocol),
        output_residual_variance=spec.output_residual_variance,
        prior_scheme="constant",
        output_likelihood="gaussian",
        output_estimator="mean",
    )
    inference = InferenceConfig(
        steps=protocol.inference_steps,
        mean_learning_rate=protocol.inference_mean_learning_rate,
        log_variance_learning_rate=protocol.inference_log_variance_learning_rate,
        initial_variance=protocol.initial_latent_variance,
        initial_mean_perturbation_std=0.0,
    )
    update = UpdateConfig(
        steps=protocol.update_steps,
        hidden_mean_learning_rate=(
            protocol.weight_mean_learning_rate if learn_hidden else 0.0
        ),
        hidden_log_variance_learning_rate=(
            protocol.weight_log_variance_learning_rate if learn_hidden else 0.0
        ),
        output_mean_learning_rate=(
            0.0 if learn_hidden else protocol.weight_mean_learning_rate
        ),
        output_log_variance_learning_rate=(
            0.0 if learn_hidden else protocol.weight_log_variance_learning_rate
        ),
        hidden_weight_kl_scale=1.0,
        output_weight_kl_scale=1.0,
        hidden_mean_kl_scale=1.0,
        output_mean_kl_scale=1.0,
        training_mc_samples=1,
        output_loss_weight=1.0,
    )
    return BPCNConfig(model=model, inference=inference, update=update)


def _build_network(
    config: BPCNConfig, spec: CellSpec, protocol: _Protocol = _PROTOCOL
) -> tuple[Network, int]:
    network = initialize_network(
        jax.random.PRNGKey(protocol.network_seed), config.model
    )
    frozen_index = 0 if spec.stage == "A_output" else 1
    frozen_layer = network.layers[frozen_index]
    identity = jnp.eye(
        frozen_layer.output_dim,
        frozen_layer.input_dim,
        dtype=frozen_layer.mean.dtype,
    )
    layers = list(network.layers)
    layers[frozen_index] = frozen_layer._replace(
        mean=identity,
        log_variance=jnp.full_like(
            frozen_layer.log_variance, MIN_WEIGHT_LOG_VARIANCE
        ),
    )
    return network._replace(layers=tuple(layers)), frozen_index


def _learned_index(stage: str) -> int:
    if stage == "A_output":
        return 1
    if stage == "B_hidden":
        return 0
    raise ValueError(f"unsupported linear-Gaussian stage: {stage!r}")


def _batches(
    data: _SyntheticData,
    target_variance: float,
    protocol: _Protocol = _PROTOCOL,
) -> tuple[Batch, ...]:
    order = np.random.default_rng(protocol.training_seed).permutation(
        protocol.training_samples
    )
    batches = []
    for start in range(0, protocol.training_samples, protocol.batch_size):
        selected = order[start : start + protocol.batch_size]
        targets = data.train_targets[selected]
        batches.append(
            Batch(
                inputs=jnp.asarray(data.train_inputs[selected], dtype=jnp.float32),
                class_indices=jnp.zeros((len(selected),), dtype=jnp.int32),
                target_mean=jnp.asarray(targets, dtype=jnp.float32),
                target_variance=jnp.full(
                    targets.shape, target_variance, dtype=jnp.float32
                ),
            )
        )
    return tuple(batches)


def _assert_frozen_layer_unchanged(
    network: Network,
    frozen_index: int,
    reference_mean: np.ndarray,
    reference_log_variance: np.ndarray,
) -> None:
    layer = network.layers[frozen_index]
    if not np.array_equal(reference_mean, np.asarray(layer.mean)):
        raise RuntimeError(f"frozen layer {frozen_index} mean changed")
    if not np.array_equal(reference_log_variance, np.asarray(layer.log_variance)):
        raise RuntimeError(f"frozen layer {frozen_index} log variance changed")


def _train_cell(
    batch_step,
    network: Network,
    spec: CellSpec,
    data: _SyntheticData,
    target_variance: float,
    protocol: _Protocol = _PROTOCOL,
) -> _TrainingOutcome:
    learned_index = _learned_index(spec.stage)
    frozen_index = 0 if spec.stage == "A_output" else 1
    frozen_reference_mean = np.asarray(network.layers[frozen_index].mean).copy()
    frozen_reference_log_variance = np.asarray(
        network.layers[frozen_index].log_variance
    ).copy()
    batches = _batches(data, target_variance, protocol)
    key = jax.random.PRNGKey(protocol.training_seed)
    history = deque(maxlen=5)
    stable_epochs = 0

    for epoch_index in range(protocol.max_epochs):
        previous_mean = np.asarray(
            network.layers[learned_index].mean, dtype=np.float64
        )
        previous_log_variance = np.asarray(
            network.layers[learned_index].log_variance, dtype=np.float64
        )
        for batch in batches:
            key, batch_key = jax.random.split(key)
            result = batch_step(network, batch, batch_key)
            network = result.network

        current_mean = np.asarray(
            network.layers[learned_index].mean, dtype=np.float64
        )
        current_log_variance = np.asarray(
            network.layers[learned_index].log_variance, dtype=np.float64
        )
        mean_delta = float(np.max(np.abs(current_mean - previous_mean)))
        log_variance_delta = float(
            np.max(np.abs(current_log_variance - previous_log_variance))
        )
        epoch = epoch_index + 1
        history.append(
            {
                "epoch": epoch,
                "mean_delta": mean_delta,
                "log_variance_delta": log_variance_delta,
            }
        )
        if not (math.isfinite(mean_delta) and math.isfinite(log_variance_delta)):
            return _TrainingOutcome(
                network=network,
                status="NON_FINITE",
                epochs=epoch,
                converged=False,
                final_mean_delta=None,
                final_log_variance_delta=None,
                history_tail=tuple(history),
            )
        if max(mean_delta, log_variance_delta) < protocol.convergence_tolerance:
            stable_epochs += 1
            if stable_epochs >= protocol.convergence_patience:
                break
        else:
            stable_epochs = 0

    _assert_frozen_layer_unchanged(
        network,
        frozen_index,
        frozen_reference_mean,
        frozen_reference_log_variance,
    )
    last = history[-1]
    converged = bool(
        max(last["mean_delta"], last["log_variance_delta"])
        < protocol.convergence_tolerance
    )
    return _TrainingOutcome(
        network=network,
        status="COMPLETE" if converged else "NOT_CONVERGED",
        epochs=int(last["epoch"]),
        converged=converged,
        final_mean_delta=float(last["mean_delta"]),
        final_log_variance_delta=float(last["log_variance_delta"]),
        history_tail=tuple(history),
    )


def _stationarity(
    config: BPCNConfig,
    batch_step,
    network: Network,
    spec: CellSpec,
    data: _SyntheticData,
    target_variance: float,
    protocol: _Protocol = _PROTOCOL,
) -> StationarityMetrics:
    learned_index = _learned_index(spec.stage)
    layer = network.layers[learned_index]
    batches = _batches(data, target_variance, protocol)
    batch_count = len(batches)

    total_mean_gradient = np.zeros_like(np.asarray(layer.mean), dtype=np.float64)
    total_log_variance_gradient = np.zeros_like(
        np.asarray(layer.log_variance), dtype=np.float64
    )
    data_mean_gradient = np.zeros_like(total_mean_gradient)
    data_log_variance_gradient = np.zeros_like(total_log_variance_gradient)
    residual_parts = []
    predictive_variance_parts = []
    implied_variance_parts = []
    exposure_parts = []

    for batch in batches:
        inferred = infer_latents(
            network,
            batch.inputs,
            batch.targets,
            config,
            jax.random.PRNGKey(protocol.training_seed),
        ).latents
        if spec.stage == "A_output":
            input_mean, input_variance = activation_moments(
                network.hidden_activations[-1],
                inferred.top_mean,
                inferred.top_variance,
            )
            target_mean, target_variance_array = (
                batch.target_mean,
                batch.target_variance,
            )
        else:
            input_mean = batch.inputs
            input_variance = jnp.zeros_like(batch.inputs)
            target_mean, target_variance_array = (
                inferred.means[0],
                inferred.variances[0],
            )

        terms = compute_layer_gradients(
            layer,
            input_mean,
            input_variance,
            target_mean,
            target_variance_array,
            data_scale=1.0,
            prior_scale=1.0 / float(batch_count),
            weight_kl_scale=1.0,
            mean_kl_scale=1.0,
        )
        total_mean_gradient += np.asarray(terms.mean_gradient, dtype=np.float64)
        total_log_variance_gradient += np.asarray(
            terms.log_variance_gradient, dtype=np.float64
        )
        data_mean_gradient += np.asarray(
            terms.mean_data_gradient, dtype=np.float64
        )
        data_log_variance_gradient += np.asarray(
            terms.log_variance_data_gradient, dtype=np.float64
        )

        residual = np.asarray(terms.variance_residual, dtype=np.float64)
        predictive_variance = np.asarray(
            terms.predictive.variance, dtype=np.float64
        )
        mean_error = np.asarray(terms.mean_error, dtype=np.float64)
        target_array = np.asarray(target_variance_array, dtype=np.float64)
        exposure = np.asarray(terms.input_second_moment, dtype=np.float64).sum(
            axis=1, keepdims=True
        ) / np.maximum(
            predictive_variance**2, np.finfo(np.float64).tiny
        )
        residual_parts.append(residual)
        predictive_variance_parts.append(predictive_variance)
        implied_variance_parts.append(target_array + mean_error**2)
        exposure_parts.append(exposure)

    moved = batch_step(
        network, batches[0], jax.random.PRNGKey(protocol.training_seed)
    ).network
    moved_layer = moved.layers[learned_index]
    mean_drift = _safe_ratio(
        float(
            np.linalg.norm(
                np.asarray(moved_layer.mean, dtype=np.float64)
                - np.asarray(layer.mean, dtype=np.float64)
            )
        ),
        float(np.linalg.norm(np.asarray(layer.mean, dtype=np.float64))),
    )
    log_variance_drift = float(
        np.max(
            np.abs(
                np.asarray(moved_layer.log_variance, dtype=np.float64)
                - np.asarray(layer.log_variance, dtype=np.float64)
            )
        )
    )
    interpretable = (
        bool(mean_drift < 0.01 and log_variance_drift < 0.01)
        if mean_drift is not None
        else None
    )

    residual = np.concatenate(residual_parts)
    predictive_variance = np.concatenate(predictive_variance_parts)
    implied_variance = np.concatenate(implied_variance_parts)
    exposure = np.concatenate(exposure_parts)
    return StationarityMetrics(
        batch_count=batch_count,
        within_batch_mean_drift=mean_drift,
        within_batch_log_variance_drift=log_variance_drift,
        residual_interpretable=interpretable,
        relative_mean_gradient_residual=_safe_ratio(
            float(np.linalg.norm(total_mean_gradient)),
            float(np.linalg.norm(data_mean_gradient)),
        ),
        relative_log_variance_gradient_residual=_safe_ratio(
            float(np.linalg.norm(total_log_variance_gradient)),
            float(np.linalg.norm(data_log_variance_gradient)),
        ),
        mean_gradient_residual_norm=float(np.linalg.norm(total_mean_gradient)),
        log_variance_gradient_residual_norm=float(
            np.linalg.norm(total_log_variance_gradient)
        ),
        mean_variance_residual=float(residual.mean()),
        mean_absolute_variance_residual=float(np.abs(residual).mean()),
        exposure_weighted_variance_residual=_safe_ratio(
            float((exposure * residual).sum()), float(exposure.sum())
        ),
        relative_exposure_weighted_variance_residual=_safe_ratio(
            float((exposure * residual).sum()),
            float((exposure * np.abs(residual)).sum()),
        ),
        predictive_variance_relation_error=_relative_error(
            predictive_variance, implied_variance
        ),
    )


def _dbpcn_predictive_nll(
    network: Network, test_inputs: np.ndarray, test_targets: np.ndarray
) -> float:
    input_mean = jnp.asarray(test_inputs, dtype=jnp.float32)
    input_variance = jnp.zeros_like(input_mean)
    predictive = moment_forward(network.layers[0], input_mean, input_variance)
    for layer_index in range(1, len(network.layers)):
        feature_mean, feature_variance = activation_moments(
            network.hidden_activations[layer_index - 1],
            predictive.mean,
            predictive.variance,
        )
        predictive = moment_forward(
            network.layers[layer_index], feature_mean, feature_variance
        )
    mean = np.asarray(predictive.mean, dtype=np.float64)
    variance = np.asarray(predictive.variance, dtype=np.float64)
    residual = np.asarray(test_targets, dtype=np.float64) - mean
    per_dimension = 0.5 * (
        np.log(2.0 * np.pi * variance) + residual**2 / variance
    )
    return float(per_dimension.sum(axis=1).mean())


def _prior_retention(
    weight_variance: np.ndarray,
    exact: _ExactPosterior,
    protocol: _Protocol,
) -> tuple[float | None, float | None, float | None]:
    unconstrained = np.isclose(np.diag(exact.gram), 0.0)
    if not unconstrained.any():
        return None, None, None
    observed = weight_variance[:, unconstrained]
    expected = np.full_like(observed, protocol.prior_std**2)
    return (
        _relative_error(observed, expected),
        float(observed.mean()),
        float(exact.diagonal_covariance[:, unconstrained].mean()),
    )


def _cell_metrics(
    network: Network,
    spec: CellSpec,
    exact: _ExactPosterior,
    data: _SyntheticData,
    protocol: _Protocol,
) -> CellMetrics:
    layer = network.layers[_learned_index(spec.stage)]
    mean = np.asarray(layer.mean, dtype=np.float64)
    weight_variance = np.asarray(layer.weight_variance(), dtype=np.float64)
    covered = np.abs(mean - exact.mean) <= 1.96 * np.sqrt(weight_variance)
    prior_error, prior_observed, prior_exact = _prior_retention(
        weight_variance, exact, protocol
    )
    return CellMetrics(
        relative_mean_error=_relative_error(mean, exact.mean),
        relative_variance_error=_relative_error(
            weight_variance, exact.diagonal_covariance
        ),
        relative_kl_optimal_variance_error=_relative_error(
            weight_variance, exact.inverse_diagonal_precision
        ),
        coverage_95=float(covered.mean()),
        mean_weight_variance=float(weight_variance.mean()),
        mean_exact_diagonal_covariance=float(exact.diagonal_covariance.mean()),
        weight_variance_ratio=_safe_ratio(
            float(weight_variance.mean()),
            float(exact.diagonal_covariance.mean()),
        ),
        total_precision_dbpcn=float(np.sum(1.0 / weight_variance)),
        total_precision_exact=float(np.trace(exact.precision)),
        dbpcn_predictive_nll=_dbpcn_predictive_nll(
            network, data.test_inputs, data.test_targets
        ),
        exact_predictive_nll=_exact_predictive_nll(
            exact, data.test_inputs, data.test_targets
        ),
        relative_prior_retention_error=prior_error,
        mean_unconstrained_weight_variance=prior_observed,
        mean_unconstrained_exact_variance=prior_exact,
    )


def _run_cell(
    spec: CellSpec,
    protocol: _Protocol = _PROTOCOL,
    batch_step=None,
) -> CellResult:
    data = _make_dataset(spec, protocol)
    noise_covariance = _noise_covariance(spec, data.true_weight, protocol)
    exact = _exact_posterior(
        data.train_inputs,
        data.train_targets,
        noise_covariance,
        protocol.prior_std,
    )
    config = _build_bpcn_config(spec, protocol)
    if batch_step is None:
        batch_step = make_batch_step(config, protocol.training_samples)
    network, _ = _build_network(config, spec, protocol)
    target_variance = _resolve_target_variance(spec, protocol)
    outcome = _train_cell(
        batch_step,
        network,
        spec,
        data,
        target_variance,
        protocol,
    )
    if outcome.status == "NON_FINITE":
        metrics = None
        stationarity = None
    else:
        metrics = _cell_metrics(outcome.network, spec, exact, data, protocol)
        stationarity = _stationarity(
            config,
            batch_step,
            outcome.network,
            spec,
            data,
            target_variance,
            protocol,
        )
    off_diagonal = noise_covariance - np.diag(np.diag(noise_covariance))
    return CellResult(
        spec=spec,
        status=outcome.status,
        converged=outcome.converged,
        epochs=outcome.epochs,
        final_mean_delta=outcome.final_mean_delta,
        final_log_variance_delta=outcome.final_log_variance_delta,
        target_variance=target_variance,
        hidden_residual_variance=_hidden_residual_variance(spec.stage, protocol),
        noise_diagonal_mean=float(np.diag(noise_covariance).mean()),
        noise_off_diagonal_max=float(np.abs(off_diagonal).max()),
        metrics=metrics,
        stationarity=stationarity,
        history_tail=outcome.history_tail,
    )


def _usability(cell: CellResult, protocol: _Protocol) -> Mapping[str, Any]:
    if cell.status == "NON_FINITE":
        return {"usable": False, "label": "NON_FINITE"}
    if cell.converged:
        return {"usable": True, "label": "CONVERGED"}
    deltas = (
        cell.final_mean_delta or 0.0,
        cell.final_log_variance_delta or 0.0,
    )
    worst_delta = max(deltas)
    trend = [
        max(item["mean_delta"], item["log_variance_delta"])
        for item in cell.history_tail
    ]
    plateaued = False
    if len(trend) >= 2 and trend[0] > 0.0:
        plateaued = (
            abs(trend[-1] - trend[0]) / trend[0]
            < protocol.plateau_relative_change
        )
    if worst_delta < protocol.numerical_floor_tolerance and plateaued:
        return {
            "usable": True,
            "label": "AT_NUMERICAL_FLOOR",
            "worst_delta": worst_delta,
        }
    return {
        "usable": False,
        "label": "STILL_MOVING",
        "worst_delta": worst_delta,
    }


def _passes(gate: str, value: float | None) -> bool | None:
    if value is None:
        return None
    definition = GATE_DEFINITIONS[gate]
    threshold = definition["threshold"]
    if definition["direction"] == "<=":
        return bool(value <= threshold)
    if definition["direction"] == "within":
        lower, upper = threshold
        return bool(lower <= value <= upper)
    raise ValueError(f"unsupported gate direction: {definition['direction']!r}")


def _evaluate_gates(
    cells: Mapping[str, CellResult], protocol: _Protocol
) -> Mapping[str, Any]:
    results = {}
    for stage in _STAGES:
        stage_results = {}
        for output_variance in protocol.output_residual_variances:
            for target_mode in _TARGET_VARIANCE_MODES:
                slot = f"b{output_variance:g}_yv_{target_mode}"

                def get(case: str) -> CellResult | None:
                    return cells.get(
                        CellSpec(stage, case, output_variance, target_mode).cell_id
                    )

                independent = get("independent")
                duplicated = get("duplicate_agree")
                collinear = get("collinear")
                entry = {}
                sources = (
                    ("G_CAL_MEAN", independent, "relative_mean_error"),
                    ("G_CAL_VAR", independent, "relative_variance_error"),
                    ("G_CAL_COVERAGE", independent, "coverage_95"),
                    (
                        "G_PRIOR_RETENTION",
                        collinear,
                        "relative_prior_retention_error",
                    ),
                )
                for gate, source, metric_name in sources:
                    value = (
                        getattr(source.metrics, metric_name)
                        if source is not None and source.metrics is not None
                        else None
                    )
                    entry[gate] = {"value": value, "passed": _passes(gate, value)}

                dbpcn_gain = None
                exact_gain = None
                gain_ratio = None
                if (
                    independent is not None
                    and duplicated is not None
                    and independent.metrics is not None
                    and duplicated.metrics is not None
                ):
                    dbpcn_gain = _safe_ratio(
                        duplicated.metrics.total_precision_dbpcn,
                        independent.metrics.total_precision_dbpcn,
                    )
                    exact_gain = _safe_ratio(
                        duplicated.metrics.total_precision_exact,
                        independent.metrics.total_precision_exact,
                    )
                    if dbpcn_gain is not None and exact_gain is not None:
                        gain_ratio = _safe_ratio(dbpcn_gain, exact_gain)
                entry["G_PREC_DIRECTION"] = {
                    "value": gain_ratio,
                    "dbpcn_precision_gain": dbpcn_gain,
                    "exact_precision_gain": exact_gain,
                    "passed": _passes("G_PREC_DIRECTION", gain_ratio),
                }
                stage_results[slot] = entry
        results[stage] = stage_results
    return results


def _classify(
    gates: Mapping[str, Any],
    cells: Mapping[str, CellResult],
    protocol: _Protocol,
) -> Mapping[str, Any]:
    usability = {
        cell_id: _usability(cell, protocol) for cell_id, cell in cells.items()
    }
    unusable = sorted(
        cell_id for cell_id, value in usability.items() if not value["usable"]
    )
    decisive_unusable = [
        cell_id for cell_id in unusable if cell_id.startswith("B_hidden")
    ]
    at_floor = sorted(
        cell_id
        for cell_id, value in usability.items()
        if value["label"] == "AT_NUMERICAL_FLOOR"
    )
    stage_b_slots = tuple(gates["B_hidden"].values())

    def all_pass(gate: str) -> bool:
        values = [slot[gate]["passed"] for slot in stage_b_slots]
        return bool(values) and all(value is True for value in values)

    mean_passed = all_pass("G_CAL_MEAN")
    variance_passed = all_pass("G_CAL_VAR")
    calibrated = mean_passed and variance_passed
    if decisive_unusable:
        status = "INCONCLUSIVE"
        summary = "At least one decisive Stage B cell is not a usable fixed point."
    elif calibrated:
        status = "CALIBRATED"
        summary = "Stage B mean and diagonal variance match the exact reference."
    elif mean_passed:
        status = "MEAN_ONLY"
        summary = "Stage B recovers the mean but not the diagonal variance."
    else:
        status = "NOT_CALIBRATED"
        summary = "Stage B does not recover the exact mean and variance reference."
    return {
        "status": status,
        "summary": summary,
        "calibration_passed": calibrated,
        "unusable_cells": unusable,
        "decisive_unusable_cells": decisive_unusable,
        "at_numerical_floor_cells": at_floor,
        "usability": usability,
    }


def _sanity_check(protocol: _Protocol = _PROTOCOL) -> Mapping[str, Any]:
    grid = _cell_grid(protocol)
    expected_count = (
        len(_STAGES)
        * len(_CASES)
        * len(protocol.output_residual_variances)
        * len(_TARGET_VARIANCE_MODES)
    )
    if len(grid) != expected_count or len({spec.cell_id for spec in grid}) != len(grid):
        raise RuntimeError("linear-Gaussian grid construction is inconsistent")
    if protocol is _PROTOCOL and len(grid) != 24:
        raise RuntimeError(f"the fixed protocol must contain 24 cells, got {len(grid)}")

    dimension = protocol.dimension
    empty_inputs = np.zeros((0, dimension), dtype=np.float64)
    empty_targets = np.zeros((0, dimension), dtype=np.float64)
    empty = _exact_posterior(
        empty_inputs,
        empty_targets,
        np.eye(dimension),
        protocol.prior_std,
    )
    prior_error = float(
        np.max(
            np.abs(empty.diagonal_covariance - protocol.prior_std**2)
        )
    )
    if prior_error > 1e-12:
        raise RuntimeError("the exact posterior does not reduce to the prior")

    probe_inputs = np.arange(2 * dimension, dtype=np.float64).reshape(2, dimension)
    covariance_a = _exact_posterior(
        probe_inputs,
        np.zeros((2, dimension)),
        np.eye(dimension),
        protocol.prior_std,
    ).covariance
    covariance_b = _exact_posterior(
        probe_inputs,
        np.ones((2, dimension)),
        np.eye(dimension),
        protocol.prior_std,
    ).covariance
    if not np.array_equal(covariance_a, covariance_b):
        raise RuntimeError("exact covariance unexpectedly depends on target values")
    return {
        "grid_cell_count": len(grid),
        "prior_reduction_max_error": prior_error,
        "exact_covariance_target_invariant": True,
    }


def run() -> LinearGaussianRunResult:
    """Run the complete fixed 24-cell protocol without saving artifacts."""
    sanity = _sanity_check(_PROTOCOL)
    grid = _cell_grid(_PROTOCOL)
    results = []
    batch_steps = {}
    for index, spec in enumerate(grid, start=1):
        step_key = (spec.stage, spec.output_residual_variance)
        if step_key not in batch_steps:
            config = _build_bpcn_config(spec, _PROTOCOL)
            batch_steps[step_key] = make_batch_step(
                config, _PROTOCOL.training_samples
            )
        cell = _run_cell(spec, _PROTOCOL, batch_steps[step_key])
        results.append(cell)
        if cell.metrics is None:
            detail = "metrics=unavailable"
        else:
            mean_error = cell.metrics.relative_mean_error
            variance_error = cell.metrics.relative_variance_error
            mean_text = "n/a" if mean_error is None else f"{mean_error:.4g}"
            variance_text = (
                "n/a" if variance_error is None else f"{variance_error:.4g}"
            )
            detail = (
                f"mean_error={mean_text} variance_error={variance_text}"
            )
        _LOG.info(
            "cell=%d/%d id=%s epochs=%d status=%s %s",
            index,
            len(grid),
            spec.cell_id,
            cell.epochs,
            cell.status,
            detail,
        )
    cells_by_id = {cell.spec.cell_id: cell for cell in results}
    gates = _evaluate_gates(cells_by_id, _PROTOCOL)
    classification = _classify(gates, cells_by_id, _PROTOCOL)
    return LinearGaussianRunResult(
        protocol=asdict(_PROTOCOL),
        sanity=sanity,
        cells=tuple(results),
        gate_definitions=GATE_DEFINITIONS,
        gates=gates,
        classification=classification,
    )


def ensure_new_output_directory(path: Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"linear-Gaussian output directory is not empty; choose a new "
            f"OUTPUT_DIR: {path}"
        )


def save_result(result: LinearGaussianRunResult, output_dir: Path) -> Path:
    """Write the experiment's only artifact: a strict JSON result."""
    output_dir = Path(output_dir)
    ensure_new_output_directory(output_dir)
    destination = output_dir / "results.json"
    atomic_write_json(
        destination,
        {
            "schema_version": _SCHEMA_VERSION,
            "experiment": "linear_gaussian",
            "protocol": result.protocol,
            "sanity": result.sanity,
            "cells": result.cells,
            "gate_definitions": result.gate_definitions,
            "gates": result.gates,
            "classification": result.classification,
        },
    )
    return destination


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_new_output_directory(OUTPUT_DIR)
    result = run()
    destination = save_result(result, OUTPUT_DIR)
    _LOG.info(
        "saved linear-Gaussian result to %s classification=%s",
        destination,
        result.classification["status"],
    )


if __name__ == "__main__":
    main()
