"""Streaming, concise metrics for BPCN training epochs."""

from dataclasses import asdict, dataclass

import numpy as np

from bpcn.training.loop import BatchStepResult


@dataclass(frozen=True)
class EpochSummary:
    batch_count: int
    initial_energy: float
    final_energy: float
    energy_delta: float
    total_energy: float
    output_energy: float
    transition_energy: float
    weight_kl: float
    mean_abs_hidden_residual: float
    max_log_variance_clamp_fraction: float
    all_finite: bool

    def to_dict(self):
        return asdict(self)


class EpochMetricsAccumulator:
    """Accumulate essential scalars without retaining per-batch records."""

    def __init__(self):
        self._batch_count = 0
        self._sums = {
            "initial_energy": 0.0,
            "final_energy": 0.0,
            "energy_delta": 0.0,
            "total_energy": 0.0,
            "output_energy": 0.0,
            "transition_energy": 0.0,
            "weight_kl": 0.0,
            "mean_abs_hidden_residual": 0.0,
        }
        self._max_clamp_fraction = 0.0
        self._all_finite = True

    @property
    def batch_count(self) -> int:
        return self._batch_count

    def add(self, result: BatchStepResult) -> None:
        e_step = result.e_step_metrics
        energy = result.energy_after
        hidden_metrics = result.layer_metrics[:-1]
        hidden_residual = (
            float(
                np.mean(
                    [
                        float(np.asarray(item.mean_abs_residual))
                        for item in hidden_metrics
                    ]
                )
            )
            if hidden_metrics
            else 0.0
        )
        values = {
            "initial_energy": float(np.asarray(e_step.initial_energy)),
            "final_energy": float(np.asarray(e_step.final_energy)),
            "energy_delta": float(
                np.asarray(e_step.final_energy - e_step.initial_energy)
            ),
            "total_energy": float(np.asarray(energy.total)),
            "output_energy": float(np.asarray(energy.output)),
            "transition_energy": float(np.asarray(energy.transitions)),
            "weight_kl": float(np.asarray(energy.weight_kl)),
            "mean_abs_hidden_residual": hidden_residual,
        }
        clamp_fraction = max(
            float(np.asarray(item.log_variance_clamp_fraction))
            for item in result.layer_metrics
        )
        finite = all(
            bool(np.asarray(item.finite)) for item in result.layer_metrics
        ) and all(np.isfinite(value) for value in values.values())
        self._batch_count += 1
        for name, value in values.items():
            self._sums[name] += value
        self._max_clamp_fraction = max(
            self._max_clamp_fraction, clamp_fraction
        )
        self._all_finite = self._all_finite and finite

    def summary(self) -> EpochSummary:
        if self._batch_count == 0:
            raise ValueError("cannot summarize an empty epoch")
        averages = {
            name: value / self._batch_count for name, value in self._sums.items()
        }
        return EpochSummary(
            batch_count=self._batch_count,
            max_log_variance_clamp_fraction=self._max_clamp_fraction,
            all_finite=self._all_finite,
            **averages,
        )
