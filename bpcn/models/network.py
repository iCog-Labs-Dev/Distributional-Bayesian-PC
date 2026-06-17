"""BPCN network: stack of L hidden Bayesian-PC layers + Bayesian output layer."""

import math
from typing import NamedTuple, Tuple
import jax

from bpcn.models.layer import Layer, init_layer


class Network(NamedTuple):
    """Diagonal Gaussian Bayesian network.

    Attributes
    ----------
    layers : tuple of Layer
        Hidden layers (length L_hidden) followed by the output layer.
    activations : tuple of str
        - "identity" : pass-through (Section 4.5 simplest case).
        - "relu"     : ReLU with delta-method moment propagation.
        - "leaky_relu": Leaky ReLU (alpha=0.01) with delta-method.
        - "tanh"     : tanh with delta-method.
    output_likelihood : str : gaussian | categorical
    """
    layers: Tuple[Layer, ...]
    activations: Tuple[str, ...] = ("identity",)
    output_likelihood: str = "gaussian"
    output_estimator: str = "mean"

    @property
    def L_hidden(self) -> int:
        return len(self.layers) - 1


# Register Network as a JAX pytree where the string-and-tuple-of-string
# fields are static auxiliary data. JAX traverses only `layers` as children
# (each Layer is itself a NamedTuple of jax arrays); `activations`,
# `output_likelihood`, `output_estimator` are part of the tree definition.
def _network_flatten(net: "Network"):
    children = (net.layers,)
    aux_data = (net.activations, net.output_likelihood, net.output_estimator)
    return children, aux_data


def _network_unflatten(aux_data, children):
    (layers,) = children
    activations, output_likelihood, output_estimator = aux_data
    return Network(
        layers=layers,
        activations=activations,
        output_likelihood=output_likelihood,
        output_estimator=output_estimator,
    )

# Register the flatten and unflatten functions with JAX. This allows JAX to treat Network as a pytree, which is necessary for
# using JAX transformations like grad, vmap, etc.
jax.tree_util.register_pytree_node(Network, _network_flatten, _network_unflatten)


def init_network(
    key: jax.Array,
    layer_dims: Tuple[int, ...],
    *,
    alpha_hidden: float = 1.0,
    alpha_output: float = 1.0,
    beta_inv_hidden: float = 1e-2,
    beta_inv_output: float = 1e-2,
    init_log_var: float = -6.0,
    hidden_init: str = "xavier",
    output_init: str = "xavier",
    activations: Tuple[str, ...] = ("identity",),
    output_likelihood: str = "gaussian",
    output_estimator: str = "mean",
    alpha_scheme: str = "constant",
    init_log_var_offset: float = 0.0,
) -> Network:
    """Initialize a network from a list of layer widths.

    layer_dims : (d_0, d_1, ..., d_L, d_y)
    alpha_hidden, alpha_output : float
        Prior scales.same across all hidden layers. Output layer can have a different scale.
    beta_inv_hidden, beta_inv_output : float
        Residual variances (v2 Eq. 24, assumption A4). Shared across hidden layers.
    init_log_var : float
    activations : tuple of str
    output_likelihood, output_estimator : str
        Output-boundary configuration
    alpha_scheme : str (default "constant")
        Weight-prior scaling scheme.
    init_log_var_offset : float (default 0.0)
        Additive offset applied to the per-HIDDEN-layer init τ after the
        `alpha_scheme` derivation. Output layer init τ is NOT offset.
    """
    if len(layer_dims) < 3:
        raise ValueError(
            f"Need at least (input, hidden_1, output) dims, got {layer_dims}"
        )
    L_hidden = len(layer_dims) - 2  # subtract input and output dims
    if len(activations) != L_hidden:
        raise ValueError(
            f"activations length ({len(activations)}) must equal L_hidden "
            f"({L_hidden}); got activations={activations}, layer_dims={layer_dims}"
        )
    _ALLOWED_ALPHA_SCHEMES = ("constant", "matched_he", "matched_xavier")
    if alpha_scheme not in _ALLOWED_ALPHA_SCHEMES:
        raise ValueError(
            f"alpha_scheme must be one of {_ALLOWED_ALPHA_SCHEMES}, got {alpha_scheme!r}"
        )

    # Derive per-layer alpha and init_log_var.
    n_layers = L_hidden + 1
    if alpha_scheme == "constant":
        layer_alphas = [float(alpha_hidden)] * L_hidden + [float(alpha_output)]
        layer_init_log_vars = [float(init_log_var)] * n_layers
    else:
        c = 2.0 if alpha_scheme == "matched_he" else 1.0
        # Linear layer i has fan_in = layer_dims[i] (the input dimensionality).
        layer_alphas = [math.sqrt(c / float(layer_dims[i])) for i in range(n_layers)]
        layer_init_log_vars = [math.log(c / float(layer_dims[i])) for i in range(n_layers)]

    # Apply init_log_var_offset to HIDDEN layers only.
    if init_log_var_offset != 0.0:
        for i in range(L_hidden):
            layer_init_log_vars[i] = layer_init_log_vars[i] + float(init_log_var_offset)

    keys = jax.random.split(key, len(layer_dims) - 1)
    layers = []
    # Hidden transitions: layers[l] maps d_l -> d_{l+1} for l = 0..L_hidden-1
    for i in range(L_hidden):
        d_in, d_out = layer_dims[i], layer_dims[i + 1]
        layers.append(
            init_layer(
                keys[i],
                d_out,
                d_in,
                alpha=layer_alphas[i],
                beta_inv=beta_inv_hidden,
                init_kind=hidden_init,
                init_log_var=layer_init_log_vars[i],
            )
        )
    # Output layer maps d_L -> d_y (layers[L_hidden])
    layers.append(
        init_layer(
            keys[-1],
            layer_dims[-1],
            layer_dims[-2],
            alpha=layer_alphas[-1],
            beta_inv=beta_inv_output,
            init_kind=output_init,
            init_log_var=layer_init_log_vars[-1],
        )
    )
    return Network(
        layers=tuple(layers),
        activations=tuple(activations),
        output_likelihood=output_likelihood,
        output_estimator=output_estimator,
    )
