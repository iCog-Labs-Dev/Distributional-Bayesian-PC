"""Validated configuration for the BPCN core."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


_ACTIVATIONS = ("identity", "relu", "leaky_relu", "tanh")
_INITIALIZERS = ("xavier", "he")
_PRIOR_SCHEMES = ("constant", "matched_he", "matched_xavier")
_OUTPUT_LIKELIHOODS = ("gaussian", "categorical")
_OUTPUT_ESTIMATORS = ("mean", "mc")


def _as_tuple(value, name: str):
    if isinstance(value, list):
        return tuple(value)
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple, got {type(value).__name__}")
    return value


def _require_finite(name: str, value: float, *, positive: bool = False) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


@dataclass(frozen=True)
class ModelConfig:
    """Network architecture, initialization, and probabilistic model settings."""

    input_dim: int = 784
    hidden_dims: Tuple[int, ...] = (128,)
    output_dim: int = 2
    activations: Tuple[str, ...] = ("relu",)

    initial_weight_log_variance: float = -6.0
    hidden_log_variance_offset: float = 0.0
    hidden_initializer: str = "xavier"
    output_initializer: str = "xavier"

    hidden_prior_std: float = 1.0
    output_prior_std: float = 1.0
    hidden_residual_variance: float = 1e-2
    output_residual_variance: float = 1e-2
    prior_scheme: str = "matched_he"

    output_likelihood: str = "categorical"
    output_estimator: str = "mc"

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dims", _as_tuple(self.hidden_dims, "hidden_dims"))
        object.__setattr__(self, "activations", _as_tuple(self.activations, "activations"))

        if self.input_dim < 1 or self.output_dim < 1:
            raise ValueError("input_dim and output_dim must be >= 1")
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")
        if len(self.activations) != len(self.hidden_dims):
            raise ValueError(
                "activations and hidden_dims must have equal length; "
                f"got {self.activations} and {self.hidden_dims}"
            )
        unknown = [name for name in self.activations if name not in _ACTIVATIONS]
        if unknown:
            raise ValueError(f"unsupported activations: {unknown}; choices: {_ACTIVATIONS}")
        if self.hidden_initializer not in _INITIALIZERS:
            raise ValueError(f"unsupported hidden_initializer: {self.hidden_initializer!r}")
        if self.output_initializer not in _INITIALIZERS:
            raise ValueError(f"unsupported output_initializer: {self.output_initializer!r}")
        if self.prior_scheme not in _PRIOR_SCHEMES:
            raise ValueError(f"unsupported prior_scheme: {self.prior_scheme!r}")
        if self.output_likelihood not in _OUTPUT_LIKELIHOODS:
            raise ValueError(f"unsupported output_likelihood: {self.output_likelihood!r}")
        if self.output_estimator not in _OUTPUT_ESTIMATORS:
            raise ValueError(f"unsupported output_estimator: {self.output_estimator!r}")

        for name in (
            "hidden_prior_std",
            "output_prior_std",
            "hidden_residual_variance",
            "output_residual_variance",
        ):
            _require_finite(name, getattr(self, name), positive=True)
        _require_finite("initial_weight_log_variance", self.initial_weight_log_variance)
        _require_finite("hidden_log_variance_offset", self.hidden_log_variance_offset)

    @property
    def layer_dims(self) -> Tuple[int, ...]:
        return (self.input_dim, *self.hidden_dims, self.output_dim)

    @property
    def hidden_layer_count(self) -> int:
        return len(self.hidden_dims)


@dataclass(frozen=True)
class InferenceConfig:
    """Latent E-step settings."""

    steps: int = 20
    mean_learning_rate: float = 0.1
    log_variance_learning_rate: float = 0.05
    initial_variance: float = 1e-2
    initial_mean_perturbation_std: float = 0.1

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        for name in (
            "mean_learning_rate",
            "log_variance_learning_rate",
            "initial_variance",
        ):
            _require_finite(name, getattr(self, name), positive=True)
        _require_finite(
            "initial_mean_perturbation_std", self.initial_mean_perturbation_std
        )
        if self.initial_mean_perturbation_std < 0:
            raise ValueError("initial_mean_perturbation_std must be >= 0")


@dataclass(frozen=True)
class UpdateConfig:
    """Weight M-step settings and objective coefficients."""

    steps: int = 16
    hidden_mean_learning_rate: float = 1e-2
    hidden_log_variance_learning_rate: float = 5e-2
    output_mean_learning_rate: float = 1e-2
    output_log_variance_learning_rate: float = 5e-3

    hidden_weight_kl_scale: float = 0.1
    output_weight_kl_scale: float = 1.0
    hidden_mean_kl_scale: Optional[float] = 0.004
    output_mean_kl_scale: Optional[float] = 0.004

    training_mc_samples: int = 1
    output_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")
        for name in (
            "hidden_mean_learning_rate",
            "hidden_log_variance_learning_rate",
            "output_mean_learning_rate",
            "output_log_variance_learning_rate",
            "hidden_weight_kl_scale",
            "output_weight_kl_scale",
        ):
            value = getattr(self, name)
            _require_finite(name, value)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        for name in ("hidden_mean_kl_scale", "output_mean_kl_scale"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value)
                if value < 0:
                    raise ValueError(f"{name} must be >= 0, got {value}")
        if self.training_mc_samples < 1:
            raise ValueError("training_mc_samples must be >= 1")
        _require_finite("output_loss_weight", self.output_loss_weight, positive=True)


@dataclass(frozen=True)
class EvaluationConfig:
    """Target-free evaluation settings."""

    inference: Optional[InferenceConfig] = None
    mc_samples: int = 16
    batch_size: int = 128

    def __post_init__(self) -> None:
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def resolve_inference(self, default: InferenceConfig) -> InferenceConfig:
        if self.inference is not None:
            return self.inference
        return InferenceConfig(
            steps=default.steps,
            mean_learning_rate=default.mean_learning_rate,
            log_variance_learning_rate=default.log_variance_learning_rate,
            initial_variance=default.initial_variance,
            initial_mean_perturbation_std=0.0,
        )


@dataclass(frozen=True)
class BPCNConfig:
    """Complete configuration required by the mathematical BPCN core."""

    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
