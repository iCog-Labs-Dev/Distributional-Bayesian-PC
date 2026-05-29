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
    psi : str
        Feature map between the hidden latent z^1 and the presynaptic
        feature into the next layer (Section 4.5). One of:
        - "identity" (default): pass-through, original linear stage.
        - "relu"   : ReLU with delta-method moment propagation
                     (Section 4.5 option 2; `relu_delta_moments` in
                     `bpcn/inference/feature_moments.py`).
        For L_hidden == 1, this is the single psi between z^1 and the
        output layer.
    output_likelihood : str
        Form of the output-boundary likelihood. See
        `categorical_output_dbpcn_continuation.pdf` for the categorical
        variant. One of:
        - "gaussian"    (default): I3 Gaussian-logit head, output term is
                        the inclusion-KL between N(y_mean, y_var) and the
                        Gaussian predictive at layer L (Eq. 12 of the
                        shared-energy note).
        - "categorical" : Bayesian softmax head
                        p(y=c|z^L, W_y) = softmax(W_y psi_L(z^L))_c
                        (Eq. 1/18 of the continuation note). Output term is
                        the categorical NLL averaged over the chosen
                        estimator (`output_estimator`).
    output_estimator : str
        Categorical F_out estimator. Only consulted when
        `output_likelihood == "categorical"`. One of:
        - "mean" : posterior-mean classifier ell^mean (Eq. 21 of continuation).
        - "mc"   : Monte Carlo categorical NLL ell^MC (Eq. 27 of continuation)
                   via reparameterized output weights and top latents.
    """
    layers: Tuple[Layer, ...]
    psi: str = "identity"
    output_likelihood: str = "gaussian"
    output_estimator: str = "mean"

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


# Register Network as a JAX pytree where the string fields (`psi`,
# `output_likelihood`, `output_estimator`) are static auxiliary data, so
# they are part of the tree definition rather than traceable leaves.
# Without this, NamedTuple's default treats `psi: str` as a leaf and JAX
# transforms (e.g. stop_gradient, jit) fail because strings aren't JAX
# types.
def _network_flatten(net: "Network"):
    children = (net.layers,)
    aux_data = (net.psi, net.output_likelihood, net.output_estimator)
    return children, aux_data


def _network_unflatten(aux_data, children):
    (layers,) = children
    psi, output_likelihood, output_estimator = aux_data
    return Network(
        layers=layers,
        psi=psi,
        output_likelihood=output_likelihood,
        output_estimator=output_estimator,
    )


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
    psi: str = "identity",
    output_likelihood: str = "gaussian",
    output_estimator: str = "mean",
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
    psi : str
        Feature map between hidden latent z^1 and the output presynaptic
        feature (Section 4.5).
    output_likelihood, output_estimator : str
        Output-boundary configuration (categorical_output_dbpcn_continuation.pdf).
        Defaults preserve the I3 Gaussian-logit head.
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
    return Network(
        layers=tuple(layers),
        psi=psi,
        output_likelihood=output_likelihood,
        output_estimator=output_estimator,
    )


def forward_mean(net: Network, x: jax.Array) -> jax.Array:
    """Pure-mean feed-forward: z^L = mu_L * ... * mu_1 * x. Used for smoke-test only.

    This does NOT do Bayesian inference. It only propagates posterior means.
    Identity psi assumed (base assumption I2).
    """
    h = x
    for layer in net.layers:
        h = h @ layer.mu.T
    return h
