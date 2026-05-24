"""Pytree utility helpers (shape assertions, positive-variance guards)."""
import jax
import jax.numpy as jnp


def shape_tree(tree):
    """Map a pytree to its shapes for debug printing."""
    return jax.tree_util.tree_map(lambda x: jnp.shape(x), tree)


def all_finite(tree) -> bool:
    """Check all leaves of a pytree are finite (no NaN/Inf)."""
    leaves = jax.tree_util.tree_leaves(tree)
    return bool(all(bool(jnp.isfinite(x).all()) for x in leaves))


def assert_positive(x, name: str = "tensor"):
    """Concrete-Python assertion that a tensor is strictly positive."""
    assert bool((jnp.asarray(x) > 0).all()), f"{name} contains non-positive values"
