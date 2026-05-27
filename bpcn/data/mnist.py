"""MNIST loader with class-incremental class filter.

Uses sklearn.datasets.fetch_openml. Caches to ~/scikit_learn_data on first call.
"""
from typing import Iterator, NamedTuple, Tuple
import numpy as np

from .moments import gaussian_logit_target


_MNIST_CACHE = {}


def _load_mnist_raw() -> Tuple[np.ndarray, np.ndarray]:
    """Load full MNIST (70000 examples). Cached in-process."""
    if "data" not in _MNIST_CACHE:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        x = ds.data.astype(np.float32) / 255.0         # [70000, 784], in [0,1]
        y = ds.target.astype(np.int64)                 # [70000]
        _MNIST_CACHE["data"] = x
        _MNIST_CACHE["target"] = y
    return _MNIST_CACHE["data"], _MNIST_CACHE["target"]


class Split(NamedTuple):
    x: np.ndarray              # [N, 784]
    y_int: np.ndarray          # [N]  original digit label (0..9)
    y_idx: np.ndarray          # [N]  mapped index into `classes` tuple


def load_split(
    classes: Tuple[int, ...],
    train: bool,
    *,
    seed: int = 0,
    n_train: int = 60000,
    n_test: int = 10000,
) -> Split:
    """Load a class-filtered MNIST split.

    The standard MNIST split is 60000 train / 10000 test. We use the first n_train
    examples as train and the last n_test as test (the openml ordering preserves
    that split when loaded with version=1).
    """
    x, y = _load_mnist_raw()
    if train:
        x, y = x[:n_train], y[:n_train]
    else:
        x, y = x[n_train:n_train + n_test], y[n_train:n_train + n_test]
    keep = np.isin(y, list(classes))
    x, y_int = x[keep], y[keep]
    # Map original digit -> index in `classes` (used for one-hot targets).
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[int(v)] for v in y_int], dtype=np.int64)
    if train:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x))
        x, y_int, y_idx = x[perm], y_int[perm], y_idx[perm]
    return Split(x=x, y_int=y_int, y_idx=y_idx)


class Batch(NamedTuple):
    x: np.ndarray             # [B, 784]
    y_idx: np.ndarray         # [B]
    y_mean: np.ndarray        # [B, C]  one-hot Gaussian target mean (I3)
    y_var: np.ndarray         # [B, C]  Gaussian target variance epsilon_y


def iter_minibatches(
    split: Split,
    *,
    batch_size: int,
    n_classes: int,
    target_var: float,
    target_scale: float = 1.0,
    drop_last: bool = True,
    shuffle_seed: int = None,
) -> Iterator[Batch]:
    """Yield batches with Gaussian-logit one-hot targets (assumption I3).

    `target_scale` sets the one-hot peak magnitude (see
    `gaussian_logit_target`); default 1.0 preserves the original encoding.
    """
    N = len(split.x)
    indices = np.arange(N)
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(indices)
    n_full = N // batch_size
    n_iters = n_full if drop_last else (n_full + (1 if N % batch_size else 0))
    for i in range(n_iters):
        sl = indices[i * batch_size:(i + 1) * batch_size]
        x = split.x[sl]
        y_idx = split.y_idx[sl]
        y_mean, y_var = gaussian_logit_target(y_idx, n_classes, target_var, target_scale)
        yield Batch(x=x, y_idx=y_idx, y_mean=y_mean, y_var=y_var)


def count_batches(split: Split, batch_size: int, drop_last: bool = True) -> int:
    N = len(split.x)
    n_full = N // batch_size
    return n_full if drop_last else (n_full + (1 if N % batch_size else 0))
