"""BPCN network: stack of L hidden Bayesian-PC layers + Gaussian-logit output layer.

References (write-up: distributional_predictive_coding_v2.pdf):
- Eq. 34: generative model p(W, Z, Y|X) = p(W) prod_n [p(y_n|z^L,W_y) prod_l p(z^l|z^{l-1},W_l)].
- Section 3.1: output likelihood Eq. 25 (Gaussian) or Eq. 26 (categorical).
- Assumption I3 (plan): one-hot Gaussian logit targets for base MNIST -> W_y treated as a regular Bayesian layer.

For base: L_hidden = 1. Hidden layer dim = 128, output dim = C.
"""
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from .layer import Layer, init_layer


class Network(NamedTuple):
    """Diagonal Gaussian Bayesian network.

    Attributes
    ----------
    layers : tuple of Layer
        Hidden layers (length L_hidden) followed by the output layer.
        layers[-1] is the output layer W_y.
    """
    layers: Tuple[Layer, ...]

    @property
    def L_hidden(self) -> int:
        return len(self.layers) - 1

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]

    @property
    def hidden_layer(self) -> Layer:
        # Convenience for L_hidden == 1.
        assert self.L_hidden == 1, "hidden_layer property assumes L_hidden == 1"
        return self.layers[0]


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
) -> Network:
    """Initialize a network from a list of layer widths.

    Parameters
    ----------
    layer_dims : (d_0, d_1, ..., d_L, d_y)
        d_0 = input dim, d_1..d_L = hidden latent dims, d_y = output dim.
    alpha_hidden, alpha_output : float
        Prior scales (Eq. 27).
    beta_inv_hidden, beta_inv_output : float
        Residual variances (Eq. 24, A4).
    init_log_var : float
        Initial tau (Eq. 99).
    """
    if len(layer_dims) < 2:
        raise ValueError("Need at least input + output dims")
    keys = jax.random.split(key, len(layer_dims) - 1)
    layers = []
    # Hidden transitions: layer l maps d_{l-1} -> d_l for l = 1..L_hidden
    L_hidden = len(layer_dims) - 2  # subtract input and output dims
    for i in range(L_hidden):
        d_in, d_out = layer_dims[i], layer_dims[i + 1]
        layers.append(
            init_layer(
                keys[i],
                d_out,
                d_in,
                alpha=alpha_hidden,
                beta_inv=beta_inv_hidden,
                init_kind=hidden_init,
                init_log_var=init_log_var,
            )
        )
    # Output layer maps d_L -> d_y
    layers.append(
        init_layer(
            keys[-1],
            layer_dims[-1],
            layer_dims[-2],
            alpha=alpha_output,
            beta_inv=beta_inv_output,
            init_kind=output_init,
            init_log_var=init_log_var,
        )
    )
    return Network(layers=tuple(layers))


def forward_mean(net: Network, x: jax.Array) -> jax.Array:
    """Pure-mean feed-forward: z^L = mu_L * ... * mu_1 * x. Used for smoke-test only.

    This does NOT do Bayesian inference. It only propagates posterior means.
    Identity psi assumed (base assumption I2).
    """
    h = x
    for layer in net.layers:
        h = h @ layer.mu.T
    return h
