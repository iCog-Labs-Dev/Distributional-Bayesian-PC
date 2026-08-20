"""Latent-state record shared by inference and training."""

from typing import NamedTuple, Tuple

import jax


class LatentState(NamedTuple):
    means: Tuple[jax.Array, ...]
    variances: Tuple[jax.Array, ...]

    @property
    def top_mean(self) -> jax.Array:
        return self.means[-1]

    @property
    def top_variance(self) -> jax.Array:
        return self.variances[-1]
