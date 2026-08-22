"""Strict, versioned checkpoint I/O for the refactored BPCN core."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from bpcn.configs.base import (
    BPCNConfig,
    EvaluationConfig,
    InferenceConfig,
    ModelConfig,
    UpdateConfig,
)
from bpcn.models.layer import Layer
from bpcn.models.network import Network


CHECKPOINT_SCHEMA_VERSION = 1
_MANIFEST_NAME = "checkpoint.json"
_WEIGHTS_NAME = "weights.npz"


class Checkpoint(NamedTuple):
    config: BPCNConfig
    network: Network
    metadata: Mapping[str, Any]


def _layer_arrays(network: Network):
    arrays = {}
    for index, layer in enumerate(network.layers):
        prefix = f"layer_{index}"
        arrays[f"{prefix}_mean"] = np.asarray(layer.mean)
        arrays[f"{prefix}_log_variance"] = np.asarray(layer.log_variance)
        arrays[f"{prefix}_residual_variance"] = np.asarray(
            layer.residual_variance
        )
        arrays[f"{prefix}_prior_std"] = np.asarray(
            layer.prior_std, dtype=np.float64
        )
    return arrays


def _validate_network(network: Network, config: BPCNConfig) -> None:
    model = config.model
    expected_layer_count = len(model.layer_dims) - 1
    if len(network.layers) != expected_layer_count:
        raise ValueError(
            f"network has {len(network.layers)} layers; expected {expected_layer_count}"
        )
    if network.hidden_activations != model.activations:
        raise ValueError("network activations do not match ModelConfig")
    if network.output_likelihood != model.output_likelihood:
        raise ValueError("network output likelihood does not match ModelConfig")
    if network.output_estimator != model.output_estimator:
        raise ValueError("network output estimator does not match ModelConfig")
    if model.prior_scheme == "constant":
        expected_priors = [model.hidden_prior_std] * model.hidden_layer_count + [
            model.output_prior_std
        ]
    else:
        coefficient = 2.0 if model.prior_scheme == "matched_he" else 1.0
        expected_priors = [
            math.sqrt(coefficient / float(model.layer_dims[index]))
            for index in range(expected_layer_count)
        ]
    for index, layer in enumerate(network.layers):
        expected_shape = (model.layer_dims[index + 1], model.layer_dims[index])
        if layer.mean.shape != expected_shape or layer.log_variance.shape != expected_shape:
            raise ValueError(
                f"layer {index} weight shape does not match {expected_shape}"
            )
        if layer.residual_variance.shape != (expected_shape[0],):
            raise ValueError(f"layer {index} residual variance has the wrong shape")
        arrays = (layer.mean, layer.log_variance, layer.residual_variance)
        if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays):
            raise ValueError(f"layer {index} contains a non-finite value")
        if np.any(np.asarray(layer.residual_variance) <= 0):
            raise ValueError(f"layer {index} residual variance must be positive")
        if not np.isfinite(layer.prior_std) or layer.prior_std <= 0:
            raise ValueError(f"layer {index} prior_std must be finite and positive")
        expected_residual = (
            model.hidden_residual_variance
            if index < model.hidden_layer_count
            else model.output_residual_variance
        )
        residual_array = np.asarray(layer.residual_variance)
        expected_residual_array = np.full(
            residual_array.shape, expected_residual, dtype=residual_array.dtype
        )
        if not np.array_equal(residual_array, expected_residual_array):
            raise ValueError(
                f"layer {index} residual variance does not match ModelConfig"
            )
        if layer.prior_std != expected_priors[index]:
            raise ValueError(f"layer {index} prior_std does not match ModelConfig")


def _atomic_json(path: Path, payload) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_checkpoint(
    directory,
    network: Network,
    config: BPCNConfig,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Atomically write a new-schema BPCN checkpoint directory."""
    _validate_network(network, config)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_payload = {} if metadata is None else dict(metadata)
    try:
        json.dumps(metadata_payload)
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint metadata must be JSON serializable") from error
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config": asdict(config),
        "metadata": metadata_payload,
    }
    _atomic_npz(destination / _WEIGHTS_NAME, _layer_arrays(network))
    _atomic_json(destination / _MANIFEST_NAME, manifest)


def _config_from_dict(payload) -> BPCNConfig:
    expected = {"model", "inference", "update", "evaluation"}
    if set(payload) != expected:
        raise ValueError(
            f"checkpoint config keys must be {sorted(expected)}, got {sorted(payload)}"
        )
    evaluation_payload = dict(payload["evaluation"])
    nested_inference = evaluation_payload.get("inference")
    if nested_inference is not None:
        evaluation_payload["inference"] = InferenceConfig(**nested_inference)
    return BPCNConfig(
        model=ModelConfig(**payload["model"]),
        inference=InferenceConfig(**payload["inference"]),
        update=UpdateConfig(**payload["update"]),
        evaluation=EvaluationConfig(**evaluation_payload),
    )


def load_checkpoint(directory) -> Checkpoint:
    """Load and validate a checkpoint produced by :func:`save_checkpoint`."""
    source = Path(directory)
    manifest_path = source / _MANIFEST_NAME
    weights_path = source / _WEIGHTS_NAME
    if not manifest_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"checkpoint requires {_MANIFEST_NAME} and {_WEIGHTS_NAME}: {source}"
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_manifest_keys = {"schema_version", "config", "metadata"}
    if set(manifest) != expected_manifest_keys:
        raise ValueError("checkpoint manifest has missing or unknown keys")
    if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported checkpoint schema version: "
            f"{manifest['schema_version']}"
        )
    config = _config_from_dict(manifest["config"])
    model = config.model
    layer_count = len(model.layer_dims) - 1
    expected_array_keys = {
        f"layer_{index}_{field}"
        for index in range(layer_count)
        for field in (
            "mean",
            "log_variance",
            "residual_variance",
            "prior_std",
        )
    }
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_array_keys:
            raise ValueError("checkpoint weights have missing or unknown arrays")
        layers = []
        for index in range(layer_count):
            prefix = f"layer_{index}"
            layers.append(
                Layer(
                    mean=jnp.asarray(archive[f"{prefix}_mean"]),
                    log_variance=jnp.asarray(
                        archive[f"{prefix}_log_variance"]
                    ),
                    residual_variance=jnp.asarray(
                        archive[f"{prefix}_residual_variance"]
                    ),
                    prior_std=float(
                        np.asarray(archive[f"{prefix}_prior_std"]).item()
                    ),
                )
            )
    network = Network(
        layers=tuple(layers),
        hidden_activations=model.activations,
        output_likelihood=model.output_likelihood,
        output_estimator=model.output_estimator,
    )
    _validate_network(network, config)
    metadata = manifest["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    return Checkpoint(config=config, network=network, metadata=metadata)
