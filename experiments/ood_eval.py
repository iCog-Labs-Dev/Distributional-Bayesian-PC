"""One-shot rotated-MNIST evaluation for one refactored MNIST run.

Edit ``MNIST_RUN_DIR`` below, then run ``python -m experiments.ood_eval``.
The source checkpoint and training history are never modified.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

from bpcn import EvaluationConfig, load_checkpoint, predict_split
from bpcn.data import DatasetSplit, load_mnist_split
from experiments._artifacts import atomic_write_json


_LOG = logging.getLogger(__name__)
_SCHEMA_VERSION = 1
_FULL_MNIST_CLASSES = tuple(range(10))
_FULL_TRAIN_SIZE = 60_000
_FULL_TEST_SIZE = 10_000
_ROTATIONS = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0)
_EVALUATION_SEED = 123
_BATCH_SIZE = 256
_MC_SAMPLES = 16
_ECE_BINS = 15
_SUPPORTED_VARIANTS = {
    "categorical_mc",
    "categorical_mean",
    "gaussian_logits",
}


# The only user-edited OOD setting.
MNIST_RUN_DIR = Path("runs/mnist_full")


@dataclass(frozen=True)
class PredictiveMetrics:
    accuracy: float
    nll: float
    entropy_mean: float
    entropy_std: float
    confidence_mean: float
    ece: float


@dataclass(frozen=True)
class AngleResult:
    angle_degrees: float
    mc: PredictiveMetrics
    mean: PredictiveMetrics
    mc_accuracy_drop: float
    mc_nll_increase: float
    mc_entropy_change: float


@dataclass(frozen=True)
class OODRunResult:
    source_run_dir: Path
    source_metadata: Mapping[str, Any]
    results: tuple[AngleResult, ...]


def expected_calibration_error(
    probabilities: np.ndarray, class_indices: np.ndarray, bins: int = _ECE_BINS
) -> float:
    """Return equal-width top-class expected calibration error."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    probabilities = np.asarray(probabilities)
    class_indices = np.asarray(class_indices)
    if probabilities.ndim != 2 or class_indices.shape != (len(probabilities),):
        raise ValueError("probabilities and class_indices have incompatible shapes")
    if len(class_indices) == 0:
        raise ValueError("cannot compute ECE for an empty dataset")
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == class_indices).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            selected = (confidence >= lower) & (confidence <= upper)
        else:
            selected = (confidence >= lower) & (confidence < upper)
        if selected.any():
            value += (
                float(selected.mean())
                * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
            )
    return float(value)


def rotate_mnist_flat(inputs: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate flattened 28x28 MNIST images without changing their support."""
    inputs = np.asarray(inputs)
    if inputs.ndim != 2 or inputs.shape[1] != 784:
        raise ValueError(f"expected flattened MNIST shape (N, 784), got {inputs.shape}")
    if float(angle_degrees) == 0.0:
        return inputs
    from scipy.ndimage import rotate as scipy_rotate

    images = inputs.reshape(len(inputs), 28, 28)
    rotated = np.empty_like(images)
    for index, image in enumerate(images):
        rotated[index] = scipy_rotate(
            image,
            float(angle_degrees),
            reshape=False,
            mode="constant",
            cval=0.0,
            order=1,
        )
    return np.clip(rotated, 0.0, 1.0).reshape(len(inputs), 784).astype(
        np.float32
    )


def _validate_source_checkpoint(checkpoint) -> None:
    metadata = checkpoint.metadata
    if metadata.get("experiment") != "mnist":
        raise ValueError("OOD source must be a refactored MNIST checkpoint")
    if metadata.get("experiment_schema_version") != 1:
        raise ValueError("unsupported or missing MNIST experiment schema version")
    variant = metadata.get("variant")
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported or missing MNIST variant: {variant!r}")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("MNIST checkpoint metadata is missing its dataset record")
    expected = {
        "name": "mnist",
        "classes": list(_FULL_MNIST_CLASSES),
        "train_size": _FULL_TRAIN_SIZE,
        "test_size": _FULL_TEST_SIZE,
        "input_dim": 784,
        "output_dim": 10,
    }
    observed = dict(dataset)
    observed["classes"] = list(observed.get("classes", ()))
    if observed != expected:
        raise ValueError(
            "OOD evaluation requires the fixed full-MNIST dataset metadata; "
            f"got {observed}"
        )
    if checkpoint.config.model.input_dim != 784:
        raise ValueError("MNIST checkpoint input dimension must be 784")
    if checkpoint.config.model.output_dim != 10:
        raise ValueError("MNIST checkpoint output dimension must be 10")


def _predictive_metrics(
    probabilities: np.ndarray, class_indices: np.ndarray
) -> PredictiveMetrics:
    probabilities = np.asarray(probabilities)
    class_indices = np.asarray(class_indices)
    if probabilities.shape != (len(class_indices), 10):
        raise ValueError(
            "predictive probabilities must have shape (sample_count, 10)"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("predictive probabilities contain a non-finite value")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("predictive probability rows do not sum to one")
    safe = np.clip(probabilities, 1e-12, 1.0)
    row_indices = np.arange(len(class_indices))
    entropy = -np.sum(safe * np.log(safe), axis=1)
    return PredictiveMetrics(
        accuracy=float((probabilities.argmax(axis=1) == class_indices).mean()),
        nll=float(-np.log(safe[row_indices, class_indices]).mean()),
        entropy_mean=float(entropy.mean()),
        entropy_std=float(entropy.std()),
        confidence_mean=float(probabilities.max(axis=1).mean()),
        ece=expected_calibration_error(
            probabilities, class_indices, bins=_ECE_BINS
        ),
    )


def _load_full_test_split(seed: int) -> DatasetSplit:
    split = load_mnist_split(
        _FULL_MNIST_CLASSES,
        train=False,
        seed=seed,
        train_size=_FULL_TRAIN_SIZE,
        test_size=_FULL_TEST_SIZE,
    )
    if len(split.inputs) != _FULL_TEST_SIZE:
        raise ValueError(
            f"full MNIST test split has {len(split.inputs)} samples; "
            f"expected {_FULL_TEST_SIZE}"
        )
    return split


def evaluate_rotations(
    checkpoint,
    split: DatasetSplit,
) -> tuple[AngleResult, ...]:
    """Evaluate fixed rotations and attach changes from the 0-degree baseline."""
    if not _ROTATIONS or float(_ROTATIONS[0]) != 0.0:
        raise RuntimeError("the first OOD rotation must be the 0-degree baseline")
    evaluation = EvaluationConfig(
        inference=checkpoint.config.evaluation.inference,
        mc_samples=_MC_SAMPLES,
        batch_size=_BATCH_SIZE,
    )
    bpcn_config = replace(checkpoint.config, evaluation=evaluation)
    raw_results = []
    for angle in _ROTATIONS:
        rotated_inputs = rotate_mnist_flat(split.inputs, angle)
        rotated_split = DatasetSplit(
            inputs=rotated_inputs,
            labels=split.labels,
            class_indices=split.class_indices,
        )
        predictions = predict_split(
            checkpoint.network,
            rotated_split,
            bpcn_config,
            jax.random.PRNGKey(_EVALUATION_SEED),
        )
        mc_metrics = _predictive_metrics(
            predictions.mc_probabilities, split.class_indices
        )
        mean_metrics = _predictive_metrics(
            predictions.mean_probabilities, split.class_indices
        )
        raw_results.append((float(angle), mc_metrics, mean_metrics))
        _LOG.info(
            "angle=%5g MC[acc=%.4f nll=%.4f H=%.4f ECE=%.4f] "
            "MEAN[acc=%.4f nll=%.4f]",
            angle,
            mc_metrics.accuracy,
            mc_metrics.nll,
            mc_metrics.entropy_mean,
            mc_metrics.ece,
            mean_metrics.accuracy,
            mean_metrics.nll,
        )

    baseline = raw_results[0][1]
    return tuple(
        AngleResult(
            angle_degrees=angle,
            mc=mc_metrics,
            mean=mean_metrics,
            mc_accuracy_drop=baseline.accuracy - mc_metrics.accuracy,
            mc_nll_increase=mc_metrics.nll - baseline.nll,
            mc_entropy_change=mc_metrics.entropy_mean - baseline.entropy_mean,
        )
        for angle, mc_metrics, mean_metrics in raw_results
    )


def run(run_dir: Path) -> OODRunResult:
    """Evaluate one refactored MNIST checkpoint on the fixed OOD protocol."""
    run_dir = Path(run_dir)
    _LOG.info("loading MNIST run from %s", run_dir)
    checkpoint = load_checkpoint(run_dir)
    _validate_source_checkpoint(checkpoint)
    split = _load_full_test_split(_EVALUATION_SEED)
    results = evaluate_rotations(checkpoint, split)
    return OODRunResult(
        source_run_dir=run_dir,
        source_metadata=checkpoint.metadata,
        results=results,
    )


def save_result(result: OODRunResult) -> Path:
    destination = result.source_run_dir / "ood_results.json"
    atomic_write_json(
        destination,
        {
            "schema_version": _SCHEMA_VERSION,
            "experiment": "rotated_mnist_ood",
            "source_run_dir": result.source_run_dir,
            "source_metadata": result.source_metadata,
            "protocol": {
                "rotations": _ROTATIONS,
                "test_size": _FULL_TEST_SIZE,
                "evaluation_seed": _EVALUATION_SEED,
                "batch_size": _BATCH_SIZE,
                "mc_samples": _MC_SAMPLES,
                "ece_bins": _ECE_BINS,
                "primary_predictive": "mc",
                "diagnostic_predictive": "posterior_mean",
            },
            "baseline": result.results[0],
            "results": result.results,
        },
    )
    return destination


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run(MNIST_RUN_DIR)
    destination = save_result(result)
    _LOG.info("saved OOD evaluation to %s", destination)


if __name__ == "__main__":
    main()
