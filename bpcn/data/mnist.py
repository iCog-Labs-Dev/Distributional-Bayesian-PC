"""MNIST loading, class filtering, and minibatch construction."""

from typing import Iterator, Tuple

import numpy as np

from bpcn.data.moments import gaussian_logit_targets
from bpcn.data.types import Batch, DatasetSplit


_MNIST_CACHE = {}


def _load_mnist_raw() -> Tuple[np.ndarray, np.ndarray]:
    if "inputs" not in _MNIST_CACHE:
        from sklearn.datasets import fetch_openml

        dataset = fetch_openml(
            "mnist_784", version=1, as_frame=False, parser="auto"
        )
        _MNIST_CACHE["inputs"] = dataset.data.astype(np.float32) / 255.0
        _MNIST_CACHE["labels"] = dataset.target.astype(np.int64)
    return _MNIST_CACHE["inputs"], _MNIST_CACHE["labels"]


def load_mnist_split(
    classes: Tuple[int, ...],
    train: bool,
    *,
    seed: int = 0,
    train_size: int = 60_000,
    test_size: int = 10_000,
) -> DatasetSplit:
    """Load the existing class-filtered OpenML MNIST split semantics."""
    classes = tuple(classes)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be a non-empty tuple of unique labels")
    if train_size < 1 or test_size < 1:
        raise ValueError("train_size and test_size must be >= 1")

    inputs, labels = _load_mnist_raw()
    if train:
        inputs, labels = inputs[:train_size], labels[:train_size]
    else:
        inputs = inputs[train_size : train_size + test_size]
        labels = labels[train_size : train_size + test_size]

    keep = np.isin(labels, classes)
    inputs = inputs[keep]
    original_labels = labels[keep]
    index_by_class = {label: index for index, label in enumerate(classes)}
    class_indices = np.asarray(
        [index_by_class[int(label)] for label in original_labels], dtype=np.int64
    )
    if train:
        permutation = np.random.default_rng(seed).permutation(len(inputs))
        inputs = inputs[permutation]
        original_labels = original_labels[permutation]
        class_indices = class_indices[permutation]
    return DatasetSplit(inputs, original_labels, class_indices)


def iter_minibatches(
    split: DatasetSplit,
    *,
    batch_size: int,
    class_count: int,
    target_variance: float,
    target_scale: float = 1.0,
    drop_last: bool = True,
    shuffle_seed: int | None = None,
) -> Iterator[Batch]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    sample_count = len(split.inputs)
    indices = np.arange(sample_count)
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(indices)

    full_batches, remainder = divmod(sample_count, batch_size)
    batch_count = full_batches if drop_last else full_batches + int(remainder > 0)
    for batch_index in range(batch_count):
        selected = indices[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]
        class_indices = split.class_indices[selected]
        target_mean, target_variance_array = gaussian_logit_targets(
            class_indices, class_count, target_variance, target_scale
        )
        yield Batch(
            inputs=split.inputs[selected],
            class_indices=class_indices,
            target_mean=target_mean,
            target_variance=target_variance_array,
        )


def count_batches(
    split: DatasetSplit, batch_size: int, drop_last: bool = True
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    full_batches, remainder = divmod(len(split.inputs), batch_size)
    return full_batches if drop_last else full_batches + int(remainder > 0)
