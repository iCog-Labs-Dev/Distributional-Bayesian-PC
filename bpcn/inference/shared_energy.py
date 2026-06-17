"""Shared distributional predictive-coding free energy."""

from typing import NamedTuple
import jax
import jax.numpy as jnp

from bpcn.models.network import Network
from bpcn.models.moments import moment_forward
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.losses.weight_kl import gaussian_weight_kl, gaussian_weight_kl_components
from bpcn.inference.feature_moments import psi_moments
from bpcn.inference.categorical_output import categorical_output_loss


class SharedEnergyTerms(NamedTuple):
    """Decomposed shared-energy components."""
    f_out: jax.Array         # scalar, per-batch mean of output local KL
    f_trans_dpc: jax.Array   # scalar, per-batch mean of hidden transition local KL
    f_weight_kl: jax.Array   # scalar, Sum_l gamma_l KL(q(W_l) || p(W_l)); NO batch averaging
    f_weight_kl_mu: jax.Array  # scalar, mu-term component of f_weight_kl
    f_weight_kl_var: jax.Array  # scalar, var-term (log-ratio) component of f_weight_kl


def _layer_presynaptic_moments(net: Network, x, m_zs, v_zs, l: int):
    """Presynaptic feature moments"""
    if l == 0:
        return x, jnp.zeros_like(x)
    return psi_moments(net.activations[l - 1], m_zs[l - 1], v_zs[l - 1])


def _hidden_layer_predictive(net: Network, x, m_zs, v_zs, l: int):
    """Predictive moments of hidden transition l
    For l == 0: predicts z^1 from x (input is deterministic).
    For l >= 1: predicts z^{l+1} from psi_l(z^l).
    """
    m_h, v_h = _layer_presynaptic_moments(net, x, m_zs, v_zs, l)
    return moment_forward(net.layers[l], m_h, v_h)


def _output_predictive(net: Network, m_l, v_l):
    """Predictive moments of the output transition (top latent only)."""
    output = net.layers[-1]
    m_h, v_h = psi_moments(net.activations[-1], m_l, v_l)
    return moment_forward(output, m_h, v_h)


def _weight_kl_total(
    net: Network,
    gamma_hidden: float,
    gamma_output: float,
    *,
    weight_kl_scale: float = 1.0,
    gamma_mu_hidden=None,
    gamma_mu_output=None,
):
    """
    Sum_l gamma_l KL(q_phi_l(W_l) || p(W_l)) over all layers l, 
    weighted by the given gamma coefficients.
    """
    L_hidden = net.L_hidden
    gmu_h = gamma_hidden if gamma_mu_hidden is None else gamma_mu_hidden
    gmu_o = gamma_output if gamma_mu_output is None else gamma_mu_output
    dtype = net.layers[0].mu.dtype
    total = jnp.zeros((), dtype=dtype)
    for l in range(L_hidden):
        layer = net.layers[l]
        mu_term, var_term = gaussian_weight_kl_components(layer.mu, layer.tau, layer.alpha)
        total = total + gmu_h * mu_term.sum() + gamma_hidden * var_term.sum()
    output = net.layers[-1]
    mu_term_o, var_term_o = gaussian_weight_kl_components(output.mu, output.tau, output.alpha)
    total = total + gmu_o * mu_term_o.sum() + gamma_output * var_term_o.sum()
    return weight_kl_scale * total


def _weight_kl_total_decomposed(
    net: Network,
    gamma_hidden: float,
    gamma_output: float,
    *,
    weight_kl_scale: float = 1.0,
    gamma_mu_hidden=None,
    gamma_mu_output=None,
):
    """Decomposed Sum_l γ_l KL(q(W_l) || p(W_l)) — mirrors `_weight_kl_total`
    but returns `(total, mu_total, var_total)`."""
    L_hidden = net.L_hidden
    gmu_h = gamma_hidden if gamma_mu_hidden is None else gamma_mu_hidden
    gmu_o = gamma_output if gamma_mu_output is None else gamma_mu_output
    dtype = net.layers[0].mu.dtype
    mu_total = jnp.zeros((), dtype=dtype)
    var_total = jnp.zeros((), dtype=dtype)
    for l in range(L_hidden):
        layer = net.layers[l]
        mu_term, var_term = gaussian_weight_kl_components(
            layer.mu, layer.tau, layer.alpha
        )
        mu_total = mu_total + gmu_h * mu_term.sum()
        var_total = var_total + gamma_hidden * var_term.sum()
    output = net.layers[-1]
    mu_term_o, var_term_o = gaussian_weight_kl_components(
        output.mu, output.tau, output.alpha
    )
    mu_total = mu_total + gmu_o * mu_term_o.sum()
    var_total = var_total + gamma_output * var_term_o.sum()
    mu_total = weight_kl_scale * mu_total
    var_total = weight_kl_scale * var_total
    return mu_total + var_total, mu_total, var_total


def _output_term(net, y_mean, y_var, y_idx, m_l, v_l, key, mc_samples_train):
    """
    Compute F_out for the active output-likelihood mode.
    Returns a scalar batch-mean F_out in per-data-point scale.
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

    Intended to be called twice per minibatch in the training loop:
      - at init time with `(e_diag.m_initial, e_diag.v_initial)` to log
        `hidden_l/init/{K,e_abs,r_abs,r_pos_frac,r_pos_mag,r_neg_mag}`,
      - at freeze time with `(frozen.m_zs, frozen.v_zs)` -- against the
        pre-M-step `net` -- to log the same metrics under `hidden_l/freeze/...`.
    Also reused by `experiments/ood_eval.py` to log per-angle wrong-subset
    residuals at the target-free fixed point.
    """
    out = {}
    for l in range(net.L_hidden):
        m_p, v_p = _hidden_layer_predictive(net, x, m_zs, v_zs, l)
        kl = gaussian_kl(m_z=m_zs[l], v_z=v_zs[l], m_p=m_p, v_p=v_p)
        r = kl.r
        r_pos = jnp.where(r > 0, r, 0)
        r_neg = jnp.where(r < 0, -r, 0)
        out[l] = {
            "K": kl.kl.mean(),
            "e_abs": jnp.abs(kl.e).mean(),
            "r_abs": jnp.abs(r).mean(),
            "r_pos_frac": (r > 0).mean().astype(kl.kl.dtype),
            "r_pos_mag": r_pos.mean(),
            "r_neg_mag": r_neg.mean(),
        }
    return out


def _trans_dpc_sum(net: Network, x, m_zs, v_zs) -> jax.Array:
    """Sum over l = 0..L_hidden-1 of the local Gaussian inclusion KLs for the hidden transitions."""
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
    gamma_mu_hidden=None,
    gamma_mu_output=None,
    weight_kl_scale: float = 1.0,
    output_weight: float = 1.0,
) -> SharedEnergyTerms:
    """Return the three decomposed terms of F_DPC"""
    f_trans_dpc = _trans_dpc_sum(net, x, m_zs, v_zs)
    # Output term consumes the top latent only (continuation Eq. 38).
    f_out_raw = _output_term(net, y_mean, y_var, y_idx, m_zs[-1], v_zs[-1], key, mc_samples_train)
    output_weight_arr = jnp.asarray(output_weight, dtype=f_out_raw.dtype)
    f_out = output_weight_arr * f_out_raw
    f_weight_kl, f_weight_kl_mu, f_weight_kl_var = _weight_kl_total_decomposed(
        net, gamma_hidden, gamma_output,
        weight_kl_scale=weight_kl_scale,
        gamma_mu_hidden=gamma_mu_hidden,
        gamma_mu_output=gamma_mu_output,
    )
    return SharedEnergyTerms(
        f_out=f_out,
        f_trans_dpc=f_trans_dpc,
        f_weight_kl=f_weight_kl,
        f_weight_kl_mu=f_weight_kl_mu,
        f_weight_kl_var=f_weight_kl_var,
    )


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
    gamma_mu_hidden=None,
    gamma_mu_output=None,
    include_weight_kl: bool = True,
    weight_kl_scale: float = 1.0,
    output_weight: float = 1.0,
) -> jax.Array:
    """F_DPC at the given (m_zs, v_zs) state, with weights held fixed.
    `m_zs, v_zs` are tuples of length `net.L_hidden`, one per hidden layer.
    The shared-KL hidden transition (extension Eq. 12 second sum) sums over
    l = 0..L_hidden-1.
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
            net, gamma_hidden, gamma_output,
            weight_kl_scale=weight_kl_scale,
            gamma_mu_hidden=gamma_mu_hidden,
            gamma_mu_output=gamma_mu_output,
        )
    return total
