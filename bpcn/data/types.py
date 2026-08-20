"""Generic data records consumed by the BPCN core."""

from typing import Any, NamedTuple

import numpy as np


class DatasetSplit(NamedTuple):
    inputs: np.ndarray
    labels: np.ndarray
    class_indices: np.ndarray


class Targets(NamedTuple):
    mean: Any
    variance: Any
    class_indices: Any


class Batch(NamedTuple):
    inputs: Any
    class_indices: Any
    target_mean: Any
    target_variance: Any

    @property
    def targets(self) -> Targets:
        return Targets(
            mean=self.target_mean,
            variance=self.target_variance,
            class_indices=self.class_indices,
        )
