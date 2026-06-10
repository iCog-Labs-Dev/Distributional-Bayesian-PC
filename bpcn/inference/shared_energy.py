"""Shared distributional predictive-coding free energy (M-SE extension).

This module implements the *canonical* shared-energy objective F_DPC of the
extension PDF at the form used throughout that write-up: Eq. 12.

References (write-up: shared_energy_dbpcn_extension.pdf):
- Eq. 12: F_DPC(lambda, phi) = F_out + Sum_l Sum_n Sum_i KL(q_lambda^l(z) || q_pred,phi^l(z))
                              + Sum_l gamma_l KL(q_phi_l(W_l) || p(W_l)).
- Eqs. 15/18: scalar local Gaussian KL (same algebra as DBPCN Eq. 65).
- Eq. 32: K_n^l = Sum_i KL(q_lambda(z_{n,i}^l) || q_pred,phi(z_{n,i}^l)).
- Eq. 72: F_DPC = F_out + F_trans-DPC + F_weight-KL (decomposition for logging).

The kappa-homotopy of extension Eq. 63 (mixing the inclusion-KL hidden
transition with the legacy Eq. 40 NLL+entropy hidden transition) was an
*optional stability mechanism* described in Section 7.1 of the extension.
It has been retired together with the legacy `pc_free_energy` objective;
this module always evaluates F_DPC at the canonical kappa=1 form (Eq. 12).

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
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from ..models.network import Network
from ..models.moments import moment_forward
from ..losses.distributional_kl import gaussian_kl
from ..losses.weight_kl import gaussian_weight_kl
from .feature_moments import psi_moments
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


def per_layer_residuals(net: Network, x, m_zs, v_zs):
    """Per-hidden-layer distributional residual diagnostics.

    For each `l in 0..net.L_hidden - 1`, computes the local Gaussian
    inclusion-KL terms used by the M-step gradient (v2 Eqs. 65/66/68;
    extension Eqs. 15/16/19) at the given (m_zs, v_zs) state, with weights
    held fixed.

    Reuses `_layer_presynaptic_moments`, `moment_forward`, and `gaussian_kl`,
    so the arithmetic is identical to `experiments/hard_subset_residuals.py
    ::_compute_residuals` and to `bpcn/training/m_step.py::update_layer` --
    just packaged as a per-batch diagnostic.

    Returns
    -------
    dict[int, dict[str, jax.Array]]
        `{l: {"K": mean per-unit KL, "e_abs": mean(|e|),
              "r_abs": mean(|r|), "r_pos_frac": mean(r > 0)}}`.
        All values are scalar JAX arrays. `K` is mean *per-unit per-example*
        (matches the `hidden_l/kl_data` convention in `LayerDiagnostics`);
        if you want the layer-sum-batch-mean form used by `_trans_dpc_sum`,
        multiply by `m_zs[l].shape[-1]`.

    Notes
    -----
    Intended to be called twice per minibatch in the training loop:
      - at init time with `(e_diag.m_initial, e_diag.v_initial)` to log
        `hidden_l/init/{K,e_abs,r_abs,r_pos_frac}`,
      - at freeze time with `(frozen.m_zs, frozen.v_zs)` -- against the
        pre-M-step `net` -- to log the same metrics under `hidden_l/freeze/...`.
    Also reused by `experiments/ood_eval.py` to log per-angle wrong-subset
    residuals at the target-free fixed point.
    """
    out = {}
    for l in range(net.L_hidden):
        m_p, v_p = _hidden_layer_predictive(net, x, m_zs, v_zs, l)
        kl = gaussian_kl(m_z=m_zs[l], v_z=v_zs[l], m_p=m_p, v_p=v_p)
        out[l] = {
            "K": kl.kl.mean(),
            "e_abs": jnp.abs(kl.e).mean(),
            "r_abs": jnp.abs(kl.r).mean(),
            "r_pos_frac": (kl.r > 0).mean().astype(kl.kl.dtype),
        }
    return out


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
    m_zs,
    v_zs,
    *,
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
    weight_kl_scale: float = 1.0,
    output_weight: float = 1.0,
) -> SharedEnergyTerms:
    """Return the three decomposed terms of F_DPC (extension Eq. 72).

    For L_hidden >= 1, `f_trans_dpc` sums over all hidden layers (extension
    Eq. 12 second sum). Sum of the three returned terms equals F_cat-DPC
    with `include_weight_kl=True` and `output_weight = lambda_y` -- i.e.
    exactly the scalar the M-step descends.

    `m_zs, v_zs` are tuples of length `net.L_hidden`, one per hidden layer.

    `output_weight` (== `lambda_y` under the categorical head, Eq. 2/48 of
    `categorical_output_dbpcn_continuation.pdf`) is folded into the returned
    `f_out` so that `f_out + f_trans_dpc + f_weight_kl` matches the descent
    scalar even when lambda_y != 1. Default 1.0 preserves the raw decomposition.

    `weight_kl_scale` rescales the returned `f_weight_kl` (see callers'
    docstrings for the convention).

    `y_idx`, `key`, `mc_samples_train` are consumed only when
    `net.output_likelihood == "categorical"`.
    """
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
    m_zs,
    v_zs,
    *,
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
    include_weight_kl: bool = True,
    weight_kl_scale: float = 1.0,
    output_weight: float = 1.0,
) -> jax.Array:
    """F_DPC of extension Eq. 12 at (lambda=(m_zs, v_zs), phi=net).

    `m_zs, v_zs` are tuples of length `net.L_hidden`, one per hidden layer.
    The shared-KL hidden transition (extension Eq. 12 second sum) sums over
    l = 0..L_hidden-1.

    Parameters
    ----------
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
        latent is anchored only by the hidden transition. Under the
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
    dtype = m_zs[0].dtype
    # Hidden transitions: inclusion-KL form (extension Eq. 12 second sum).
    hidden_term = _trans_dpc_sum(net, x, m_zs, v_zs)
    # Output boundary term: Gaussian inclusion-KL or categorical softmax NLL
    # (continuation Eq. 2/48). Operates on the *top* latent only
    # (continuation Eq. 38).
    F_out = _output_term(net, y_mean, y_var, y_idx, m_zs[-1], v_zs[-1], key, mc_samples_train)

    output_weight_arr = jnp.asarray(output_weight, dtype=dtype)
    total = output_weight_arr * F_out + hidden_term
    if include_weight_kl:
        total = total + _weight_kl_total(
            net, gamma_hidden, gamma_output, weight_kl_scale=weight_kl_scale
        )
    return total
