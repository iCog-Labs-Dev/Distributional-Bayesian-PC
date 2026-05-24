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
- F_out         : KL(N(y_mean, y_var) || q_pred,output(z_y)) summed over output
                  units, mean over batch. Uses (y_mean, y_var) Gaussian-logit
                  targets (assumption I3 of distributional_predictive_coding_v2.pdf
                  Section 8.2 Option 2). Same target encoding as the existing
                  M-step output update in bpcn/training/m_step.py:m_step.
- F_trans_dpc   : KL(N(m_l, v_l) || q_pred,hidden(z^1)) summed over hidden units,
                  mean over batch.
- F_weight_kl   : Sum_l gamma_l Sum_{ij} KL(q_phi_l(w_ij) || p(w_ij)) using Eq. 77.

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
from .free_energy import transition_neg_log_density, latent_neg_entropy


class SharedEnergyTerms(NamedTuple):
    """Decomposed shared-energy components (extension Eq. 72)."""
    f_out: jax.Array         # scalar, per-batch mean of output local KL
    f_trans_dpc: jax.Array   # scalar, per-batch mean of hidden transition local KL
    f_weight_kl: jax.Array   # scalar, Sum_l gamma_l KL(q(W_l) || p(W_l)); NO batch averaging


def _hidden_predictive(net: Network, x):
    """Predictive moments of the hidden transition. Eq. 94 sets v_h^0 = 0."""
    hidden = net.layers[0]
    return moment_forward(hidden, x, jnp.zeros_like(x))


def _output_predictive(net: Network, m_l, v_l):
    """Predictive moments of the output transition (identity psi_1, I2)."""
    output = net.layers[1]
    return moment_forward(output, m_l, v_l)


def _weight_kl_total(
    net: Network,
    gamma_hidden: float,
    gamma_output: float,
    *,
    weight_kl_scale: float = 1.0,
):
    """Sum_l gamma_l KL(q_phi_l(W_l) || p(W_l)). Constant w.r.t. latents.

    `weight_kl_scale` rescales the full-data weight KL to whatever convention
    the caller's data terms are written in. Default 1.0 = the extension Eq. 12
    full-data scale. Pass 1/N_train when the data terms have already been
    averaged over the batch (per-data-point scale), so that the assembled
    F_DPC matches the per-data-point objective the M-step's gradient
    (data_scale=1/B, prior_scale=1/N) actually descends.
    """
    hidden = net.layers[0]
    output = net.layers[1]
    Kw_h = gaussian_weight_kl(hidden.mu, hidden.tau, hidden.alpha).sum()
    Kw_o = gaussian_weight_kl(output.mu, output.tau, output.alpha).sum()
    return weight_kl_scale * (gamma_hidden * Kw_h + gamma_output * Kw_o)


def shared_energy_terms(
    net: Network,
    x: jax.Array,
    y_mean: jax.Array,
    y_var: jax.Array,
    m_l: jax.Array,
    v_l: jax.Array,
    *,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
    weight_kl_scale: float = 1.0,
) -> SharedEnergyTerms:
    """Return the three decomposed terms of F_DPC at kappa=1 (extension Eq. 72).

    Sum of the three equals F_DPC at kappa=1 with include_weight_kl=True.
    Used for diagnostics and unit tests (S10/S12).

    `weight_kl_scale` rescales the returned `f_weight_kl` so the sum
    (f_out + f_trans_dpc + f_weight_kl) is on a consistent scale. The data
    terms (f_out, f_trans_dpc) are batch *means* (per-data-point scale), so
    callers that want the per-data-point F_DPC -- the scalar the M-step's
    gradient with data_scale=1/B, prior_scale=1/N actually descends -- should
    pass `weight_kl_scale = 1.0 / N_train`. Default 1.0 leaves the weight KL
    on the full-data scale of extension Eq. 12, which mixes scales when added
    to the batch-averaged data terms and is intended only for unit tests that
    pair it with a matched M-step (prior_scale=1).
    """
    assert net.L_hidden == 1, "shared_energy_terms assumes L_hidden == 1 (base BPCN)"
    m_p1, v_p1 = _hidden_predictive(net, x)
    f_trans_dpc = gaussian_kl(m_z=m_l, v_z=v_l, m_p=m_p1, v_p=v_p1).kl.sum(axis=-1).mean()
    m_py, v_py = _output_predictive(net, m_l, v_l)
    f_out = gaussian_kl(m_z=y_mean, v_z=y_var, m_p=m_py, v_p=v_py).kl.sum(axis=-1).mean()
    f_weight_kl = _weight_kl_total(
        net, gamma_hidden, gamma_output, weight_kl_scale=weight_kl_scale
    )
    return SharedEnergyTerms(f_out=f_out, f_trans_dpc=f_trans_dpc, f_weight_kl=f_weight_kl)


def shared_free_energy(
    net: Network,
    x: jax.Array,
    y_mean: jax.Array,
    y_var: jax.Array,
    m_l: jax.Array,
    v_l: jax.Array,
    *,
    kappa=1.0,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
    include_weight_kl: bool = True,
    weight_kl_scale: float = 1.0,
) -> jax.Array:
    """F_kappa from extension Eq. 63 at (lambda=(m_l, v_l), phi=net).

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
    """
    assert net.L_hidden == 1, "shared_free_energy assumes L_hidden == 1 (base BPCN)"
    hidden = net.layers[0]
    # Hidden transition: shared-KL form.
    m_p1, v_p1 = _hidden_predictive(net, x)
    K_hidden_dpc = gaussian_kl(m_z=m_l, v_z=v_l, m_p=m_p1, v_p=v_p1).kl.sum(axis=-1).mean()
    # Hidden transition: legacy NLL + entropy form (only used at kappa<1).
    nll_1 = transition_neg_log_density(
        m_l, v_l, m_h=x, v_h=jnp.zeros_like(x), layer=hidden
    ).mean()
    neg_H = latent_neg_entropy(v_l).mean()
    kappa_arr = jnp.asarray(kappa, dtype=m_l.dtype)
    hidden_term = (1.0 - kappa_arr) * (nll_1 + neg_H) + kappa_arr * K_hidden_dpc
    # Output transition: always shared-KL form, not annealed (extension Eq. 63).
    m_py, v_py = _output_predictive(net, m_l, v_l)
    K_out = gaussian_kl(m_z=y_mean, v_z=y_var, m_p=m_py, v_p=v_py).kl.sum(axis=-1).mean()

    total = K_out + hidden_term
    if include_weight_kl:
        total = total + _weight_kl_total(
            net, gamma_hidden, gamma_output, weight_kl_scale=weight_kl_scale
        )
    return total
