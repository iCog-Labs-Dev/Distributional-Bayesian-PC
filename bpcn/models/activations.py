"""Activation evaluation and delta-method moment propagation."""

import jax
import jax.numpy as jnp


LEAKY_RELU_SLOPE = 0.01


def _identity_moments(mean, variance):
    return mean, variance


def _relu_moments(mean, variance):
    active = (mean > 0).astype(mean.dtype)
    return active * mean, active * variance


def _leaky_relu_moments(mean, variance):
    active = (mean > 0).astype(mean.dtype)
    slope = active + LEAKY_RELU_SLOPE * (1.0 - active)
    return slope * mean, slope**2 * variance


def _tanh_moments(mean, variance):
    transformed = jnp.tanh(mean)
    derivative = 1.0 - transformed**2
    return transformed, derivative**2 * variance


_MOMENT_FUNCTIONS = {
    "identity": _identity_moments,
    "relu": _relu_moments,
    "leaky_relu": _leaky_relu_moments,
    "tanh": _tanh_moments,
}


def activation_moments(name: str, mean, variance):
    """Propagate diagonal moments through the configured activation."""
    try:
        function = _MOMENT_FUNCTIONS[name]
    except KeyError as error:
        raise ValueError(
            f"unsupported activation: {name!r}; choices: {tuple(_MOMENT_FUNCTIONS)}"
        ) from error
    return function(mean, variance)


def apply_activation_sample(name: str, value):
    if name == "identity":
        return value
    if name == "relu":
        return jax.nn.relu(value)
    if name == "leaky_relu":
        return jax.nn.leaky_relu(value, negative_slope=LEAKY_RELU_SLOPE)
    if name == "tanh":
        return jnp.tanh(value)
    raise ValueError(f"unsupported activation: {name!r}")
