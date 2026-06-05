"""Shared distributional predictive-coding free energy (M-SE extension).

References (write-up: shared_energy_dbpcn_extension.pdf):
- Eq. 12: F_DPC(lambda, phi) = F_out + Sum_l Sum_n Sum_i KL(q_lambda^l(z) || q_pred,phi^l(z))
                              + Sum_l gamma_l KL(q_phi_l(W_l) || p(W_l)).
- Eqs. 15/18: scalar local Gaussian KL (same algebra as DBPCN Eq. 65).
- Eq. 32: K_n^l = Sum_i KL(q_lambda(z_{n,i}^l) || q_pred,phi(z_{n,i}^l)).
- Eq. 63: kappa-homotopy between original Eq. 40 hidden-transition (NLL + entropy)
          and the shared inclusion-KL hidden-transition. Output term and weight KL
          are NOT annealed.
- Eq. 72: F_DPC = F_out + F_trans-DPC + F_weight-KL (decomposition for logging).

For the base BPCN (L_hidden == 1) this decomposes as:
- F_out         : Output-boundary term. Form depends on
                  `net.output_likelihood`:
                    * "gaussian"    -> KL(N(y_mean, y_var) || q_pred,output(z_y))
                                       summed over output units, mean over batch
                                       (assumption I3 of v2 Section 8.2 Option 2).
                    * "categorical" -> softmax NLL via MEAN (Eq. 21 of
                                       categorical_output_dbpcn_continuation.pdf)
                                       or MC (Eq. 27) on the integer-class target
                                       `y_idx`. F_cat-DPC = lambda_y F_out +
                                       F_trans-DPC + F_weight-KL (continuation
                                       Eq. 2/48).
- F_trans_dpc   : KL(N(m_l, v_l) || q_pred,hidden(z^1)) summed over hidden units,
                  mean over batch. Unchanged across output-likelihood modes.
- F_weight_kl   : Sum_l gamma_l Sum_{ij} KL(q_phi_l(w_ij) || p(w_ij)) using Eq. 77
                  of v2 (== Eq. 44 of the continuation note for the output head;
                  algebraically identical).

Backward compatibility note. At kappa=0 the hidden-transition term becomes
(NLL + neg-entropy) -- matching the *hidden* part of the legacy Eq. 40
free energy but NOT the output part. To recover legacy F_z (Eq. 40) exactly,
dispatch to bpcn.inference.free_energy.free_energy via the E-step
`objective` parameter; do NOT use this module with kappa=0 alone.

Numerical note. kappa is treated as a traced scalar inside jit (we always
evaluate both the legacy NLL+entropy branch and the shared-KL branch and
blend by kappa). This keeps the compiled graph constant across epochs as
kappa anneals.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from ..models.network import Network
from ..models.moments import moment_forward
from ..losses.distributional_kl import gaussian_kl
from ..losses.weight_kl import gaussian_weight_kl
from .feature_moments import psi_moments
from .free_energy import transition_neg_log_density, latent_neg_entropy
from .categorical_output import categorical_output_loss


class SharedEnergyTerms(NamedTuple):
    """Decomposed shared-energy components (extension Eq. 72)."""
    f_out: jax.Array         # scalar, per-batch mean of output local KL
    f_trans_dpc: jax.Array   # scalar, per-batch mean of hidden transition local KL
    f_weight_kl: jax.Array   # scalar, Sum_l gamma_l KL(q(W_l) || p(W_l)); NO batch averaging


def _layer_presynaptic_moments(net: Network, x, m_zs, v_zs, l: int):
    """Presynaptic feature moments (m_h, v_h) feeding into `net.layers[l]`.

    For l == 0 the presynaptic is the deterministic input (v2 Eq. 94: v_h^0 = 0).
    For l >= 1 the presynaptic is psi_{l-1}(z^l) via the delta-method
    moment transform (v2 Eq. 21 / Section 4.5).
    """
    if l == 0:
        return x, jnp.zeros_like(x)
    return psi_moments(net.activations[l - 1], m_zs[l - 1], v_zs[l - 1])


def _hidden_layer_predictive(net: Network, x, m_zs, v_zs, l: int):
    """Predictive moments of hidden transition l (v2 Eqs. 60-61 / extension Eqs. 10-11).

    For l == 0: predicts z^1 from x (input is deterministic).
    For l >= 1: predicts z^{l+1} from psi_l(z^l).
    """
    m_h, v_h = _layer_presynaptic_moments(net, x, m_zs, v_zs, l)
    return moment_forward(net.layers[l], m_h, v_h)


def _output_predictive(net: Network, m_l, v_l):
    """Predictive moments of the output transition (top latent only).

    The presynaptic feature for the output layer is psi_L(z^L); moments
    propagate via `psi_moments(net.activations[-1], m_l, v_l)`
    (v2 Eq. 21 / continuation Section 4). Here (m_l, v_l) are the *top*
    hidden latent's frozen moments.
    """
    output = net.layers[-1]
    m_h, v_h = psi_moments(net.activations[-1], m_l, v_l)
    return moment_forward(output, m_h, v_h)


def _weight_kl_total(
    net: Network,
    gamma_hidden: float,
    gamma_output: float,
    *,
    weight_kl_scale: float = 1.0,
):
    """Sum_l gamma_l KL(q_phi_l(W_l) || p(W_l)) (extension Eq. 12 third sum).

    For multi-layer support this loops over all hidden layers (one γ_hidden
    coefficient shared per user clarification) and adds the output-layer
    weight KL with γ_output. Constant w.r.t. latents.

    `weight_kl_scale` rescales the full-data weight KL to whatever convention
    the caller's data terms are written in. Default 1.0 = the extension Eq. 12
    full-data scale. Pass 1/N_train when the data terms have already been
    averaged over the batch (per-data-point scale), so that the assembled
    F_DPC matches the per-data-point objective the M-step's gradient
    (data_scale=1/B, prior_scale=1/N) actually descends.
    """
    L_hidden = net.L_hidden
    Kw_hidden = jnp.zeros((), dtype=net.layers[0].mu.dtype)
    for l in range(L_hidden):
        layer = net.layers[l]
        Kw_hidden = Kw_hidden + gaussian_weight_kl(layer.mu, layer.tau, layer.alpha).sum()
    output = net.layers[-1]
    Kw_o = gaussian_weight_kl(output.mu, output.tau, output.alpha).sum()
    return weight_kl_scale * (gamma_hidden * Kw_hidden + gamma_output * Kw_o)


def _output_term(net, y_mean, y_var, y_idx, m_l, v_l, key, mc_samples_train):
    """Compute F_out for the active output-likelihood mode.

    Returns a scalar batch-mean F_out in per-data-point scale.

    - Gaussian (legacy): inclusion-KL between N(y_mean, y_var) and the
      Gaussian predictive at layer L (Eq. 12 of M-SE extension; assumption
      I3 of v2 Section 8.2 Option 2).
    - Categorical (continuation Eq. 2/48): MEAN or MC softmax NLL on the
      integer-class target `y_idx`. `key` and `mc_samples_train` are only
      consulted under the MC estimator; passing dummy values when MEAN is
      harmless (the dispatcher ignores them).
    """
    if net.output_likelihood == "gaussian":
        m_py, v_py = _output_predictive(net, m_l, v_l)
        return gaussian_kl(m_z=y_mean, v_z=y_var, m_p=m_py, v_p=v_py).kl.sum(axis=-1).mean()
    elif net.output_likelihood == "categorical":
        return categorical_output_loss(net, m_l, v_l, y_idx, key, mc_samples_train)
    else:
        raise ValueError(
            f"unknown output_likelihood: {net.output_likelihood!r}; "
            f"choices: 'gaussian', 'categorical'"
        )


def _coerce_latent_tuples(m_l, v_l):
    """Accept either a single array (legacy L=1 callers) or a tuple of arrays.

    Multi-layer callers pass `(m_0, m_1, ...)`, length L_hidden. Legacy L=1
    callers pass a single array; we wrap it in a 1-tuple so the rest of the
    function can treat both uniformly. The returned tuples are always length
    L_hidden.
    """
    m_zs = m_l if isinstance(m_l, tuple) else (m_l,)
    v_zs = v_l if isinstance(v_l, tuple) else (v_l,)
    return m_zs, v_zs


def _trans_dpc_sum(net: Network, x, m_zs, v_zs) -> jax.Array:
    """Sum over l = 0..L_hidden-1 of the local Gaussian inclusion KL
    (extension Eq. 12 second sum; extension Eqs. 13-15).

    Each layer contributes `KL(N(m_zs[l], v_zs[l]) || N(m_p^l, v_p^l))`
    summed over postsynaptic units and averaged over the batch.
    """
    total = jnp.zeros((), dtype=m_zs[0].dtype)
    for l in range(net.L_hidden):
        m_p, v_p = _hidden_layer_predictive(net, x, m_zs, v_zs, l)
        total = total + gaussian_kl(
            m_z=m_zs[l], v_z=v_zs[l], m_p=m_p, v_p=v_p,
        ).kl.sum(axis=-1).mean()
    return total


def shared_energy_terms(
    net: Network,
    x: jax.Array,
    y_mean: jax.Array,
    y_var: jax.Array,
    m_l,
    v_l,
    *,
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
    weight_kl_scale: float = 1.0,
    output_weight: float = 1.0,
) -> SharedEnergyTerms:
    """Return the three decomposed terms of F_DPC at kappa=1 (extension Eq. 72).

    For L_hidden >= 1, `f_trans_dpc` sums over all hidden layers (extension
    Eq. 12 second sum). Sum of the three returned terms equals F_cat-DPC at
    kappa=1 with `include_weight_kl=True` and `output_weight = lambda_y` --
    i.e. exactly the scalar the M-step descends.

    `m_l, v_l` may be either a single array (legacy L=1 callers) or a tuple
    of length L_hidden (multi-layer). `_coerce_latent_tuples` normalises the
    input.

    `output_weight` (== `lambda_y` under the categorical head, Eq. 2/48 of
    `categorical_output_dbpcn_continuation.pdf`) is folded into the returned
    `f_out` so that `f_out + f_trans_dpc + f_weight_kl` matches the descent
    scalar even when lambda_y != 1. Default 1.0 preserves the raw decomposition.

    `weight_kl_scale` rescales the returned `f_weight_kl` (see callers'
    docstrings for the convention).

    `y_idx`, `key`, `mc_samples_train` are consumed only when
    `net.output_likelihood == "categorical"`.
    """
    m_zs, v_zs = _coerce_latent_tuples(m_l, v_l)
    f_trans_dpc = _trans_dpc_sum(net, x, m_zs, v_zs)
    # Output term consumes the top latent only (continuation Eq. 38).
    f_out_raw = _output_term(net, y_mean, y_var, y_idx, m_zs[-1], v_zs[-1], key, mc_samples_train)
    output_weight_arr = jnp.asarray(output_weight, dtype=f_out_raw.dtype)
    f_out = output_weight_arr * f_out_raw
    f_weight_kl = _weight_kl_total(
        net, gamma_hidden, gamma_output, weight_kl_scale=weight_kl_scale
    )
    return SharedEnergyTerms(f_out=f_out, f_trans_dpc=f_trans_dpc, f_weight_kl=f_weight_kl)


def shared_free_energy(
    net: Network,
    x: jax.Array,
    y_mean: jax.Array,
    y_var: jax.Array,
    m_l,
    v_l,
    *,
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
    kappa=1.0,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
    include_weight_kl: bool = True,
    weight_kl_scale: float = 1.0,
    output_weight: float = 1.0,
) -> jax.Array:
    """F_kappa from extension Eq. 63 at (lambda=(m_l, v_l), phi=net).

    `m_l, v_l` may be either a single array (legacy L=1 callers) or a tuple
    of length L_hidden (multi-layer). All hidden-layer terms — both the
    shared-KL branch (extension Eq. 12) and the legacy NLL+entropy branch
    used at kappa < 1 — sum over l = 0..L_hidden-1.

    Parameters
    ----------
    kappa : float or jax scalar in [0, 1]
        kappa=1 -> pure shared energy. Hidden transition is the local Gaussian KL.
        kappa=0 -> hidden transition is the ordinary expected NLL + (-H) from Eq. 40.
        Output term and weight KL are not homotopized.
        Always traced as a jnp scalar so the jit graph is constant in kappa.
    gamma_hidden, gamma_output : float
        Per-layer KL-to-prior coefficients (Eq. 62 gamma_l). Forwarded to the
        weight KL term.
    include_weight_kl : bool (static)
        If False, the weight-KL sum is omitted. The E-step caller should pass
        False because that term is constant w.r.t. latents and including it
        only adds compile cost.
    weight_kl_scale : float
        Multiplier on the weight-KL term when `include_weight_kl=True`. See
        `shared_energy_terms` docstring for the convention: pass 1/N_train to
        match the per-data-point M-step gradient scale; default 1.0 = the
        full-data convention of extension Eq. 12 (used by S11/S13 paired with
        m_step(prior_scale=1)). Ignored when `include_weight_kl=False`.
    output_weight : float
        Multiplier on F_out (the output-boundary term). 1.0 = full
        observed target (extension Section 4 / Algorithm 1); 0.0 = target-free
        test-time inference (v2 write-up Section 6.6 paragraph 1) where the
        latent is anchored only by the hidden transition. Parallel to the
        `output_weight` kwarg in legacy `free_energy.free_energy`. Under the
        categorical output head this multiplier plays the role of `lambda_y`
        (Eq. 2/48 of `categorical_output_dbpcn_continuation.pdf`): set it
        from `cfg.lambda_y` to use a non-1.0 task weight.
    y_idx, key, mc_samples_train
        Categorical-mode kwargs. Ignored when `net.output_likelihood ==
        "gaussian"`. Under `"categorical"`, `y_idx` (shape [B], int) is the
        target class index; `key` is the PRNG key for the MC estimator;
        `mc_samples_train` is the static number of MC samples. The MEAN
        estimator ignores `key` and `mc_samples_train`.
    """
    m_zs, v_zs = _coerce_latent_tuples(m_l, v_l)
    dtype = m_zs[0].dtype
    # Hidden transitions: shared-KL form summed over all hidden layers
    # (extension Eq. 12 second sum).
    K_hidden_dpc = _trans_dpc_sum(net, x, m_zs, v_zs)
    # Hidden transitions: legacy NLL + entropy form summed over all hidden
    # layers (v2 Eq. 40 applied per layer). Only used at kappa < 1.
    nll_sum = jnp.zeros((), dtype=dtype)
    neg_H_sum = jnp.zeros((), dtype=dtype)
    for l in range(net.L_hidden):
        m_h, v_h = _layer_presynaptic_moments(net, x, m_zs, v_zs, l)
        nll_sum = nll_sum + transition_neg_log_density(
            m_zs[l], v_zs[l], m_h=m_h, v_h=v_h, layer=net.layers[l]
        ).mean()
        neg_H_sum = neg_H_sum + latent_neg_entropy(v_zs[l]).mean()
    kappa_arr = jnp.asarray(kappa, dtype=dtype)
    hidden_term = (1.0 - kappa_arr) * (nll_sum + neg_H_sum) + kappa_arr * K_hidden_dpc
    # Output boundary term: Gaussian inclusion-KL (legacy) or categorical
    # softmax NLL (continuation Eq. 2/48). Operates on the *top* latent only
    # (continuation Eq. 38). Not annealed by kappa (extension Eq. 63).
    F_out = _output_term(net, y_mean, y_var, y_idx, m_zs[-1], v_zs[-1], key, mc_samples_train)

    output_weight_arr = jnp.asarray(output_weight, dtype=dtype)
    total = output_weight_arr * F_out + hidden_term
    if include_weight_kl:
        total = total + _weight_kl_total(
            net, gamma_hidden, gamma_output, weight_kl_scale=weight_kl_scale
        )
    return total
