"""One-shot full-MNIST experiment for the refactored BPCN core.

Edit the settings block below, then run ``python -m experiments.mnist``.
There is intentionally no command-line configuration surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from bpcn import (
    BPCNConfig,
    Batch,
    EpochMetricsAccumulator,
    EvaluationConfig,
    EvaluationMetrics,
    InferenceConfig,
    ModelConfig,
    Network,
    UpdateConfig,
    evaluate_split,
    initialize_network,
    make_batch_step,
    save_checkpoint,
)
from bpcn.data import DatasetSplit, iter_minibatches, load_mnist_split
from bpcn.training import EpochSummary
from experiments._artifacts import atomic_write_json


_LOG = logging.getLogger(__name__)
_FULL_MNIST_CLASSES = tuple(range(10))
_FULL_TRAIN_SIZE = 60_000
_FULL_TEST_SIZE = 10_000
_INPUT_DIM = 784
_OUTPUT_DIM = 10
_SUPPORTED_ACTIVATIONS = {"identity", "relu", "leaky_relu", "tanh"}
_EXPERIMENT_SCHEMA_VERSION = 1


class MNISTVariant(str, Enum):
    CATEGORICAL_MC = "categorical_mc"
    CATEGORICAL_MEAN = "categorical_mean"
    GAUSSIAN_LOGITS = "gaussian_logits"


@dataclass(frozen=True)
class MNISTExperimentConfig:
    """Editable full-MNIST protocol and grouped BPCN optimization settings."""

    variant: MNISTVariant = MNISTVariant.CATEGORICAL_MC
    mlp_layers: tuple[tuple[int, str], ...] = ((128, "relu"),)
    batch_size: int = 128
    epochs: int = 5
    evaluate_every: int = 1
    seed: int = 0
    evaluation_mc_samples: int = 16
    run_dir: Path = Path("runs/mnist_full")
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    update: UpdateConfig = field(default_factory=UpdateConfig)
    gaussian_target_variance: float = 1e-3
    gaussian_target_scale: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.variant, str):
            object.__setattr__(self, "variant", MNISTVariant(self.variant))
        layers = tuple(self.mlp_layers)
        if not layers:
            raise ValueError("mlp_layers must contain at least one hidden layer")
        normalized = []
        for index, layer in enumerate(layers):
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                raise TypeError(
                    "each MLP layer must be a (neuron_count, activation) pair"
                )
            width, activation = layer
            if isinstance(width, bool) or not isinstance(width, int) or width < 1:
                raise ValueError(
                    f"MLP layer {index} neuron count must be a positive integer"
                )
            if activation not in _SUPPORTED_ACTIVATIONS:
                raise ValueError(
                    f"MLP layer {index} uses unsupported activation {activation!r}"
                )
            normalized.append((width, activation))
        object.__setattr__(self, "mlp_layers", tuple(normalized))
        object.__setattr__(self, "run_dir", Path(self.run_dir))

        for name in (
            "batch_size",
            "epochs",
            "evaluate_every",
            "evaluation_mc_samples",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.gaussian_target_variance <= 0:
            raise ValueError("gaussian_target_variance must be > 0")
        if self.gaussian_target_scale <= 0:
            raise ValueError("gaussian_target_scale must be > 0")


# ---------------------------------------------------------------------------
# Editable experiment settings
# ---------------------------------------------------------------------------

MNIST_VARIANT = MNISTVariant.CATEGORICAL_MC

MLP_LAYERS = (
    (128, "relu"),
)

BATCH_SIZE = 128
EPOCHS = 5
EVALUATE_EVERY = 1
SEED = 0
EVALUATION_MC_SAMPLES = 16
RUN_DIR = Path("runs/mnist_full")

INFERENCE = InferenceConfig()
UPDATE = UpdateConfig()

EXPERIMENT = MNISTExperimentConfig(
    variant=MNIST_VARIANT,
    mlp_layers=MLP_LAYERS,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    evaluate_every=EVALUATE_EVERY,
    seed=SEED,
    evaluation_mc_samples=EVALUATION_MC_SAMPLES,
    run_dir=RUN_DIR,
    inference=INFERENCE,
    update=UPDATE,
)


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any] | None


@dataclass(frozen=True)
class MNISTRunResult:
    network: Network
    bpcn_config: BPCNConfig
    epochs: tuple[EpochRecord, ...]
    final_evaluation: EvaluationMetrics
    train_sample_count: int
    test_sample_count: int


def resolve_bpcn_config(settings: MNISTExperimentConfig) -> BPCNConfig:
    """Resolve the selected preset into one internally consistent BPCN config."""
    likelihood, estimator = {
        MNISTVariant.CATEGORICAL_MC: ("categorical", "mc"),
        MNISTVariant.CATEGORICAL_MEAN: ("categorical", "mean"),
        MNISTVariant.GAUSSIAN_LOGITS: ("gaussian", "mean"),
    }[settings.variant]
    hidden_dims = tuple(width for width, _ in settings.mlp_layers)
    activations = tuple(activation for _, activation in settings.mlp_layers)
    return BPCNConfig(
        model=ModelConfig(
            input_dim=_INPUT_DIM,
            hidden_dims=hidden_dims,
            output_dim=_OUTPUT_DIM,
            activations=activations,
            output_likelihood=likelihood,
            output_estimator=estimator,
        ),
        inference=settings.inference,
        update=settings.update,
        evaluation=EvaluationConfig(
            mc_samples=settings.evaluation_mc_samples,
            batch_size=settings.batch_size,
        ),
    )


def load_full_mnist(seed: int) -> tuple[DatasetSplit, DatasetSplit]:
    """Load the fixed full 10-class MNIST train/test protocol."""
    train = load_mnist_split(
        _FULL_MNIST_CLASSES,
        train=True,
        seed=seed,
        train_size=_FULL_TRAIN_SIZE,
        test_size=_FULL_TEST_SIZE,
    )
    test = load_mnist_split(
        _FULL_MNIST_CLASSES,
        train=False,
        seed=seed,
        train_size=_FULL_TRAIN_SIZE,
        test_size=_FULL_TEST_SIZE,
    )
    if len(train.inputs) != _FULL_TRAIN_SIZE:
        raise ValueError(
            f"full MNIST training split has {len(train.inputs)} samples; "
            f"expected {_FULL_TRAIN_SIZE}"
        )
    if len(test.inputs) != _FULL_TEST_SIZE:
        raise ValueError(
            f"full MNIST test split has {len(test.inputs)} samples; "
            f"expected {_FULL_TEST_SIZE}"
        )
    return train, test


def _device_batch(batch: Batch) -> Batch:
    return Batch(
        inputs=jnp.asarray(batch.inputs),
        class_indices=jnp.asarray(batch.class_indices),
        target_mean=jnp.asarray(batch.target_mean),
        target_variance=jnp.asarray(batch.target_variance),
    )


def train_epoch(
    network: Network,
    split: DatasetSplit,
    settings: MNISTExperimentConfig,
    batch_step,
    epoch: int,
    training_key,
) -> tuple[Network, EpochSummary]:
    """Train one shuffled epoch and return only the essential summary."""
    if len(split.inputs) < settings.batch_size:
        raise ValueError("training split is smaller than one full minibatch")
    accumulator = EpochMetricsAccumulator()
    epoch_key = jax.random.fold_in(training_key, epoch)
    for batch_index, batch in enumerate(
        iter_minibatches(
            split,
            batch_size=settings.batch_size,
            class_count=_OUTPUT_DIM,
            target_variance=settings.gaussian_target_variance,
            target_scale=settings.gaussian_target_scale,
            drop_last=True,
            shuffle_seed=settings.seed + epoch,
        )
    ):
        result = batch_step(
            network,
            _device_batch(batch),
            jax.random.fold_in(epoch_key, batch_index),
        )
        network = result.network
        accumulator.add(result)
    return network, accumulator.summary()


def evaluate_network(
    network: Network,
    split: DatasetSplit,
    bpcn_config: BPCNConfig,
    key,
) -> EvaluationMetrics:
    return evaluate_split(network, split, bpcn_config, key)


def _evaluation_dict(metrics: EvaluationMetrics) -> dict[str, Any]:
    return dict(metrics._asdict())


def run(settings: MNISTExperimentConfig) -> MNISTRunResult:
    """Run the configured full-MNIST experiment without saving artifacts."""
    bpcn_config = resolve_bpcn_config(settings)
    _LOG.info("loading full MNIST")
    train_split, test_split = load_full_mnist(settings.seed)
    root_key = jax.random.PRNGKey(settings.seed)
    initialization_key, training_key, evaluation_key = jax.random.split(root_key, 3)
    network = initialize_network(initialization_key, bpcn_config.model)
    batch_step = make_batch_step(bpcn_config, len(train_split.inputs))
    records = []
    final_evaluation = None

    _LOG.info(
        "training variant=%s architecture=%s train=%d test=%d",
        settings.variant.value,
        settings.mlp_layers,
        len(train_split.inputs),
        len(test_split.inputs),
    )
    for epoch in range(1, settings.epochs + 1):
        network, summary = train_epoch(
            network,
            train_split,
            settings,
            batch_step,
            epoch,
            training_key,
        )
        evaluation = None
        if epoch % settings.evaluate_every == 0 or epoch == settings.epochs:
            final_evaluation = evaluate_network(
                network,
                test_split,
                bpcn_config,
                jax.random.fold_in(evaluation_key, epoch),
            )
            evaluation = _evaluation_dict(final_evaluation)
            _LOG.info(
                "epoch=%d/%d energy=%.5f MC[acc=%.4f nll=%.4f] "
                "MEAN[acc=%.4f nll=%.4f]",
                epoch,
                settings.epochs,
                summary.total_energy,
                final_evaluation.mc_accuracy,
                -final_evaluation.mc_log_likelihood,
                final_evaluation.mean_accuracy,
                -final_evaluation.mean_log_likelihood,
            )
        else:
            _LOG.info(
                "epoch=%d/%d energy=%.5f delta=%.5f finite=%s",
                epoch,
                settings.epochs,
                summary.total_energy,
                summary.energy_delta,
                summary.all_finite,
            )
        records.append(
            EpochRecord(
                epoch=epoch,
                training=summary.to_dict(),
                evaluation=evaluation,
            )
        )

    if final_evaluation is None:
        raise RuntimeError("the final MNIST epoch was not evaluated")
    return MNISTRunResult(
        network=network,
        bpcn_config=bpcn_config,
        epochs=tuple(records),
        final_evaluation=final_evaluation,
        train_sample_count=len(train_split.inputs),
        test_sample_count=len(test_split.inputs),
    )


def ensure_new_run_directory(path: Path) -> None:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"MNIST run directory is not empty; choose a new RUN_DIR: {path}"
        )


def save_run(settings: MNISTExperimentConfig, result: MNISTRunResult) -> Path:
    """Save one new-format checkpoint and concise JSON history."""
    ensure_new_run_directory(settings.run_dir)
    final_metrics = _evaluation_dict(result.final_evaluation)
    metadata = {
        "experiment": "mnist",
        "experiment_schema_version": _EXPERIMENT_SCHEMA_VERSION,
        "variant": settings.variant.value,
        "dataset": {
            "name": "mnist",
            "classes": _FULL_MNIST_CLASSES,
            "train_size": result.train_sample_count,
            "test_size": result.test_sample_count,
            "input_dim": _INPUT_DIM,
            "output_dim": _OUTPUT_DIM,
        },
        "architecture": {"mlp_layers": settings.mlp_layers},
        "training": {
            "batch_size": settings.batch_size,
            "epochs": settings.epochs,
            "evaluate_every": settings.evaluate_every,
            "seed": settings.seed,
            "evaluation_mc_samples": settings.evaluation_mc_samples,
            "gaussian_target_variance": settings.gaussian_target_variance,
            "gaussian_target_scale": settings.gaussian_target_scale,
        },
        "final_evaluation": final_metrics,
    }
    save_checkpoint(
        settings.run_dir,
        result.network,
        result.bpcn_config,
        metadata=metadata,
    )
    atomic_write_json(
        settings.run_dir / "history.json",
        {
            "schema_version": _EXPERIMENT_SCHEMA_VERSION,
            "experiment": "mnist",
            "variant": settings.variant,
            "epochs": result.epochs,
            "final_evaluation": final_metrics,
        },
    )
    return settings.run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_new_run_directory(EXPERIMENT.run_dir)
    result = run(EXPERIMENT)
    destination = save_run(EXPERIMENT, result)
    _LOG.info("saved MNIST run to %s", destination)


if __name__ == "__main__":
    main()
