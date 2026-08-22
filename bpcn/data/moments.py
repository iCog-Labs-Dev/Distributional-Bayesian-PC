"""Target-moment construction helpers."""

from typing import Tuple

import numpy as np


def gaussian_logit_targets(
    class_indices: np.ndarray,
    class_count: int,
    target_variance: float,
    target_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode class indices as scaled one-hot Gaussian target moments."""
    if class_count < 1:
        raise ValueError("class_count must be >= 1")
    if target_variance <= 0:
        raise ValueError("target_variance must be > 0")
    if target_scale <= 0:
        raise ValueError("target_scale must be > 0")
    indices = np.asarray(class_indices)
    if indices.ndim != 1:
        raise ValueError("class_indices must be one-dimensional")
    if np.any(indices < 0) or np.any(indices >= class_count):
        raise ValueError("class_indices contain a value outside the class range")

    target_mean = np.zeros((indices.shape[0], class_count), dtype=np.float32)
    target_mean[np.arange(indices.shape[0]), indices] = float(target_scale)
    target_var = np.full(
        (indices.shape[0], class_count), float(target_variance), dtype=np.float32
    )
    return target_mean, target_var
