"""Algorithm 3: BPCN training loop driver.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 6.5 Algorithm 3.
- Section 6.7 stopping / diagnostics.

References (extension: shared_energy_dbpcn_extension.pdf):
- Section 6.3 Algorithm 3: shared-energy training loop.
- Eq. 63: kappa-homotopy per epoch.

The per-batch step is jit-compiled with the hyperparameter constants closed
over so JAX can fuse the E-step and M-step into a single graph (they still
do NOT share gradients -- see plan section 6.5). The `kappa` argument is
dynamic so it can change per epoch without retracing.
"""
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp

from ..inference.e_step import e_step
from ..inference.shared_energy import shared_energy_terms, shared_free_energy
from ..models.network import Network
from ..models.layer import Layer
from ..utils.safe_math import clamp_tau
from .m_step import m_step


def _with_loop_kl_summary(first_diag, final_diag):
    """Attach full inner-loop KL movement to the final-step diagnostics."""
    return final_diag._replace(
        kl_data_loop_before=first_diag.kl_data_before,
        kl_data_loop_after=final_diag.kl_data_after,
        kl_data_loop_delta=final_diag.kl_data_after - first_diag.kl_data_before,
    )


def make_batch_step(cfg, N_train: int):
    """Build a jit-compiled function batch_step(net, x, y_mean, y_var, y_idx, key, kappa) -> BatchOutcome.

    Uses the SGD-per-data-point scaling convention (see training/m_step.py
    docstring):
        data_scale  = 1/B
        prior_scale = 1/N_train
    so that eta values live in the conventional 1e-3 range. Mathematically
    equivalent to Eq. 81 with eta rescaled by 1/N_train.

    The `kappa` runtime argument enables per-epoch kappa-homotopy (extension
    Eq. 63) without re-tracing the compiled graph. When `cfg.objective ==
    'pc_free_energy'` kappa is ignored.
    """
    data_scale = 1.0 / float(cfg.batch_size)
    prior_scale = 1.0 / float(N_train)
    # F_DPC diagnostic and accept_or_damp gate must use the same per-data-point
    # scale as the M-step's gradient. Data terms f_out/f_trans_dpc are already
    # batch means (1/B); the weight KL gets the matching 1/N_train multiplier.
    # See bpcn/inference/shared_energy.py weight_kl_scale docstring.
    weight_kl_scale = 1.0 / float(N_train)

    m_step_iters = int(cfg.m_step_iters)
    if m_step_iters < 1:
        raise ValueError(f"m_step_iters must be >= 1, got {m_step_iters}")

    # Capture cfg constants in the closure so JAX treats them as static.
    objective = cfg.objective
    rho_z = float(cfg.rho_z)
    rho_w = float(cfg.rho_w)
    r_max = cfg.r_max
    gamma_hidden = float(cfg.gamma_hidden)
    gamma_output = float(cfg.gamma_output)
    accept_or_damp = bool(cfg.accept_or_damp)
    accept_damp_omega = float(cfg.accept_damp_omega)
    accept_damp_tol = float(cfg.accept_damp_tol)
    # Categorical-output extension constants (continuation note Eq. 2/48).
    mc_samples_train = int(cfg.mc_samples_train)
    lambda_y = float(cfg.lambda_y)

    @partial(jax.jit, static_argnums=())
    def batch_step(net, x, y_mean, y_var, y_idx, key, kappa):
        # Split the batch key into (E-step key, M-step keys, accept-or-damp
        # key). M-step inner iters and accept-or-damp gate each get fresh
        # randomness independent of the E-step's MC samples.
        n_keys = 2 + m_step_iters
        keys = jax.random.split(key, n_keys)
        key_estep = keys[0]
        keys_mstep = keys[1:1 + m_step_iters]
        key_gate = keys[-1]
        # `output_weight=lambda_y` keeps the E-step's output-term weight in
        # sync with the M-step's `lambda_y * F_out` (continuation Eq. 2/48).
        # Under Gaussian mode lambda_y defaults to 1.0 so legacy behaviour
        # is unchanged.
        frozen, e_diag = e_step(
            net, x, y_mean,
            T_z=cfg.T_z,
            eta_m=cfg.eta_m,
            eta_u=cfg.eta_u,
            v_init=cfg.v_init,
            output_weight=lambda_y,
            y_var=y_var,
            objective=objective,
            kappa=kappa,
            rho_z=rho_z,
            gamma_hidden=gamma_hidden,
            gamma_output=gamma_output,
            y_idx=y_idx,
            key=key_estep,
            mc_samples_train=mc_samples_train,
        )
        # M-step: run m_step_iters sequential gradient updates on the SAME
        # frozen latent posterior. Each iteration recomputes (m_p, v_p) from
        # the current (mu, tau) so the gradient signal updates as the weights
        # move. Section 6.7 explicitly permits "one or a few" inner M-step
        # gradient updates per minibatch. The Python loop is unrolled at jit
        # trace time so the compiled graph contains m_step_iters m_step calls.
        # `net_old` is the start-of-batch network used as the prox reference
        # (extension Eq. 64). Only matters when rho_w > 0.
        net_old = net
        new_net = net
        first_m_diag = None
        m_diag = None
        for it in range(m_step_iters):
            new_net, m_diag = m_step(
                new_net, frozen, x, y_mean, y_var,
                alpha_hidden=cfg.alpha_hidden,
                alpha_output=cfg.alpha_output,
                gamma_hidden=cfg.gamma_hidden,
                gamma_output=cfg.gamma_output,
                eta_mu_hidden=cfg.eta_mu_hidden,
                eta_tau_hidden=cfg.eta_tau_hidden,
                eta_mu_output=cfg.eta_mu_output,
                eta_tau_output=cfg.eta_tau_output,
                data_scale=data_scale,
                prior_scale=prior_scale,
                r_max=r_max,
                rho_w=rho_w,
                net_old=net_old,
                y_idx=y_idx,
                key=keys_mstep[it],
                mc_samples_train=mc_samples_train,
                lambda_y=lambda_y,
            )
            if first_m_diag is None:
                first_m_diag = m_diag
        # Compose per-layer first/last diagnostics across the m_step_iters
        # inner loop. For L_hidden == 1 the hidden diag key is "hidden"
        # (legacy); for L >= 2 it's "hidden_0", "hidden_1", ... — see
        # `m_step` in bpcn/training/m_step.py for the naming convention.
        m_diag = {
            k: _with_loop_kl_summary(first_m_diag[k], m_diag[k])
            for k in m_diag
        }

        # Accept-or-damp post-M-step check (extension Eq. 68). Interpolates
        # new_net toward net_old if F_DPC grew beyond tolerance. The trigger
        # flag isn't returned (would require a 5-tuple); enable verbose log
        # via post-hoc analysis of saved F_DPC trace if needed.
        if accept_or_damp:
            new_net = _accept_or_damp(
                net_old, new_net, x, y_mean, y_var, frozen,
                kappa=kappa,
                gamma_hidden=gamma_hidden, gamma_output=gamma_output,
                omega=accept_damp_omega, tol=accept_damp_tol,
                weight_kl_scale=weight_kl_scale,
                output_weight=lambda_y,
                y_idx=y_idx,
                key=key_gate,
                mc_samples_train=mc_samples_train,
            )

        # F_DPC decomposition snapshot AFTER M-step (extension Eq. 72).
        # `output_weight=lambda_y` makes the returned f_out match the scalar
        # the M-step descended, so f_out + f_trans_dpc + f_weight_kl reads as
        # F_cat-DPC. Reuse the gate key for the MC diagnostic so the
        # comparison the gate uses and the snapshot we log come from the
        # same RNG draw.
        # Pass the *tuple* of per-layer latents (frozen.m_zs / frozen.v_zs)
        # so the multi-layer F_trans_dpc sum visits every hidden layer.
        # Under L_hidden==1 these are 1-tuples and the result matches the
        # legacy single-array call exactly.
        f_dpc_terms = shared_energy_terms(
            new_net, x, y_mean, y_var, frozen.m_zs, frozen.v_zs,
            y_idx=y_idx, key=key_gate, mc_samples_train=mc_samples_train,
            gamma_hidden=gamma_hidden, gamma_output=gamma_output,
            weight_kl_scale=weight_kl_scale,
            output_weight=lambda_y,
        )
        return new_net, e_diag, m_diag, f_dpc_terms

    return batch_step


def _interpolate_network(net_old, net_new, omega):
    """Linear interpolation network <- (1-omega) net_old + omega net_new (extension Eq. 68).

    Linear in (mu, tau) space; tau is re-clamped after blending. beta_inv and
    alpha are taken from net_new (they're configuration constants). Static
    aux fields (psi, output_likelihood, output_estimator) propagate from
    net_new -- they must agree across both arguments for accept-or-damp to
    be meaningful, so taking either is equivalent.
    """
    blended = []
    for old_l, new_l in zip(net_old.layers, net_new.layers):
        mu_blend = (1.0 - omega) * old_l.mu + omega * new_l.mu
        tau_blend = clamp_tau((1.0 - omega) * old_l.tau + omega * new_l.tau)
        blended.append(Layer(
            mu=mu_blend, tau=tau_blend,
            beta_inv=new_l.beta_inv, alpha=new_l.alpha,
        ))
    return Network(
        layers=tuple(blended),
        activations=net_new.activations,
        output_likelihood=net_new.output_likelihood,
        output_estimator=net_new.output_estimator,
    )


def _accept_or_damp(net_old, net_new, x, y_mean, y_var, frozen, *,
                    kappa, gamma_hidden, gamma_output, omega, tol,
                    weight_kl_scale,
                    y_idx=None, key=None, mc_samples_train: int = 1,
                    output_weight: float = 1.0):
    """Compare F_DPC(phi_new) to F_DPC(phi_old); interpolate if it grew (extension Eq. 68).

    Returns the chosen (net_new if F_DPC decreased or stayed within tol,
    else linearly interpolated network). `weight_kl_scale` must match the
    M-step's prior_scale convention (1/N_train under the production
    per-data-point setup) so the gate measures the same scalar the M-step
    descended. `output_weight` must equal `cfg.lambda_y` (the multiplier the
    categorical M-step applies to F_out per continuation Eq. 2/48) so the
    gate measures the same F_cat-DPC that descent operates on; default 1.0
    matches Gaussian-mode behaviour. Under the categorical head the same
    `key` is fed to both f_dpc calls so the comparison is over the same MC
    sample draw.
    """
    def f_dpc(net_):
        # Multi-layer aware: pass per-layer latent tuples so F_trans_dpc
        # sums correctly when L_hidden >= 2.
        return shared_free_energy(
            net_, x, y_mean, y_var, frozen.m_zs, frozen.v_zs,
            y_idx=y_idx, key=key, mc_samples_train=mc_samples_train,
            kappa=kappa, gamma_hidden=gamma_hidden, gamma_output=gamma_output,
            include_weight_kl=True, weight_kl_scale=weight_kl_scale,
            output_weight=output_weight,
        )
    f_old = f_dpc(net_old)
    f_new = f_dpc(net_new)
    triggered = (f_new - f_old) > tol
    chosen = jax.lax.cond(
        triggered,
        lambda _: _interpolate_network(net_old, net_new, omega),
        lambda _: net_new,
        operand=None,
    )
    return chosen
