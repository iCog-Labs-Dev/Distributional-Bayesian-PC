"""RNG key splitting helpers.

Single global root key; explicit jax.random.split per stage/epoch/minibatch.
No implicit RNG state.
"""
from typing import Tuple
import jax


def split(key, n: int = 2) -> Tuple[jax.Array, ...]:
    """Split a PRNGKey into n subkeys."""
    return tuple(jax.random.split(key, n))


def fold(key, data: int) -> jax.Array:
    """Fold an integer into a key (useful for per-epoch / per-batch keys)."""
    return jax.random.fold_in(key, data)
