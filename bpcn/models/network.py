"""BPCN network: stack of L hidden Bayesian-PC layers + Bayesian output layer.

References (write-up: distributional_predictive_coding_v2.pdf):
- Eq. 7: hierarchy z^0, z^1, ..., z^L with z^0 = x clamped.
- Eq. 17 / Eq. 34: generative model
    p(W, Z, Y|X) = p(W) prod_n [p(y_n | z^L, W_y) prod_l p(z^l | z^{l-1}, W_l)].
- Eq. 21 / Section 4.5: presynaptic feature h^{l-1} = psi_{l-1}(z^{l-1});
  per-layer activations propagate moments through `psi_moments`.
- Section 3.1: output likelihood Eq. 25 (Gaussian) or Eq. 26 (categorical).

For multi-layer support: `Network.activations` is a tuple of length L_hidden.
`activations[l]` is the feature map applied to z^{l+1} (output of layer l)
before it enters layer l+1. With L_hidden == 1 this collapses to a single
psi between the one hidden latent and the output, matching the legacy
single-layer architecture.
"""
from typing import NamedTuple, Tuple
import jax

from .layer import Layer, init_layer


class Network(NamedTuple):
    """Diagonal Gaussian Bayesian network.

    Attributes
    ----------
    layers : tuple of Layer
        Hidden layers (length L_hidden) followed by the output layer.
        `layers[-1]` is the output layer W_y. Length is L_hidden + 1.
    activations : tuple of str
        Per-layer feature maps psi_l between consecutive layers
        (Section 4.5; v2 Eq. 21). Length equals L_hidden. `activations[l]`
        is the activation applied to the output of `layers[l]` before it
        is fed into `layers[l+1]`. For L_hidden == 1, this is the single
        psi between the one hidden latent and the output layer. Choices:
        - "identity" : pass-through (Section 4.5 simplest case).
        - "relu"     : ReLU with delta-method moment propagation.
        - "leaky_relu": Leaky ReLU (alpha=0.01) with delta-method.
        - "tanh"     : tanh with delta-method.
    output_likelihood : str
        Form of the output-boundary likelihood. See
        `categorical_output_dbpcn_continuation.pdf` for the categorical
        variant. One of:
        - "gaussian"    (default): I3 Gaussian-logit head, output term is
                        the inclusion-KL between N(y_mean, y_var) and the
                        Gaussian predictive at the top layer (extension Eq. 12).
        - "categorical" : Bayesian softmax head
                        p(y=c|z^L, W_y) = softmax(W_y psi_L(z^L))_c
                        (continuation Eq. 1/18). Output term is the
                        categorical NLL averaged over `output_estimator`.
    output_estimator : str
        Categorical F_out estimator. Only consulted when
        `output_likelihood == "categorical"`. One of:
        - "mean" : posterior-mean classifier ell^mean (continuation Eq. 21).
        - "mc"   : Monte Carlo categorical NLL ell^MC (continuation Eq. 27)
                   via reparameterized output weights and top latents.
    """
    layers: Tuple[Layer, ...]
    activations: Tuple[str, ...] = ("identity",)
    output_likelihood: str = "gaussian"
    output_estimator: str = "mean"

    @property
    def L_hidden(self) -> int:
        return len(self.layers) - 1

    @property
    def output_layer(self) -> Layer:
        return self.layers[-1]


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
) -> Network:
    """Initialize a network from a list of layer widths.

    Parameters
    ----------
    layer_dims : (d_0, d_1, ..., d_L, d_y)
        d_0 = input dim, d_1..d_L = hidden latent dims, d_y = output dim.
        L_hidden = len(layer_dims) - 2.
    alpha_hidden, alpha_output : float
        Prior scales (v2 Eq. 27). Shared across all hidden layers.
    beta_inv_hidden, beta_inv_output : float
        Residual variances (v2 Eq. 24, assumption A4). Shared across hidden layers.
    init_log_var : float
        Initial tau = log sigma_0^2 (v2 Eq. 99).
    activations : tuple of str
        Per-layer feature maps psi_l (Section 4.5; v2 Eq. 21). Must have
        length == L_hidden. Each entry is one of
        {"identity", "relu", "leaky_relu", "tanh"}.
    output_likelihood, output_estimator : str
        Output-boundary configuration (categorical_output_dbpcn_continuation.pdf).
        Defaults preserve the I3 Gaussian-logit head.
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
                alpha=alpha_hidden,
                beta_inv=beta_inv_hidden,
                init_kind=hidden_init,
                init_log_var=init_log_var,
            )
        )
    # Output layer maps d_L -> d_y (layers[L_hidden])
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
        activations=tuple(activations),
        output_likelihood=output_likelihood,
        output_estimator=output_estimator,
    )
