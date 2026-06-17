"""
BPCN training loop driver.

The per-batch step is jit-compiled with the hyperparameter constants closed
over so JAX can fuse the E-step and M-step into a single graph (they still
do NOT share gradients.
"""
from functools import partial
import jax

from bpcn.inference.e_step import e_step
from bpcn.inference.shared_energy import per_layer_residuals, shared_energy_terms
from bpcn.training.m_step import m_step


def _with_loop_kl_summary(first_diag, final_diag):
    """Attach full inner-loop KL movement to the final-step diagnostics."""
    return final_diag._replace(
        kl_data_loop_before=first_diag.kl_data_before,
        kl_data_loop_after=final_diag.kl_data_after,
        kl_data_loop_delta=final_diag.kl_data_after - first_diag.kl_data_before,
    )


def make_batch_step(cfg, N_train: int):
    """
    Build a jit-compiled function
    batch_step(net, x, y_mean, y_var, y_idx, key) -> BatchOutcome.
    Uses the SGD-per-data-point scaling convention
    """
    data_scale = 1.0 / float(cfg.batch_size)
    prior_scale = 1.0 / float(N_train)
    weight_kl_scale = 1.0 / float(N_train)

    m_step_iters = int(cfg.m_step_iters)
    if m_step_iters < 1:
        raise ValueError(f"m_step_iters must be >= 1, got {m_step_iters}")

    # Capture cfg constants in the closure so JAX treats them as static.
    gamma_hidden = float(cfg.gamma_hidden)
    gamma_output = float(cfg.gamma_output)
    # Decoupled γ_μ overrides
    gamma_mu_hidden = (
        float(cfg.gamma_mu_hidden) if cfg.gamma_mu_hidden is not None else None
    )
    gamma_mu_output = (
        float(cfg.gamma_mu_output) if cfg.gamma_mu_output is not None else None
    )
    # Categorical-output extension constants (continuation note Eq. 2/48).
    mc_samples_train = int(cfg.mc_samples_train)
    lambda_y = float(cfg.lambda_y)
    # Predictive-disequilibrium init coefficient
    init_perturb_std = float(cfg.init_perturb_std)

    @partial(jax.jit, static_argnums=())
    def batch_step(net, x, y_mean, y_var, y_idx, key):
        # Split the batch key into (E-step key, M-step keys, F_DPC-snapshot key).
        n_keys = 2 + m_step_iters
        keys = jax.random.split(key, n_keys)
        key_estep = keys[0]
        keys_mstep = keys[1:1 + m_step_iters]
        key_snapshot = keys[-1]

        frozen, e_diag = e_step(
            net, x, y_mean,
            T_z=cfg.T_z,
            eta_m=cfg.eta_m,
            eta_u=cfg.eta_u,
            v_init=cfg.v_init,
            output_weight=lambda_y,
            y_var=y_var,
            gamma_hidden=gamma_hidden,
            gamma_output=gamma_output,
            y_idx=y_idx,
            key=key_estep,
            mc_samples_train=mc_samples_train,
            init_perturb_std=init_perturb_std,
        )

        # Init-time per-layer residuals: K^l, |e|, |r|, r_pos_frac evaluated at
        # the (possibly perturbed) starting state of the E-step, BEFORE any
        # latent descent step.
        init_residuals = per_layer_residuals(
            net, x, e_diag.m_initial, e_diag.v_initial,
        )
        # Freeze-time per-layer residuals.
        freeze_residuals = per_layer_residuals(
            net, x, frozen.m_zs, frozen.v_zs,
        )
        # M-step: run m_step_iters sequential gradient updates on the SAME
        # frozen latent posterior.
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
                y_idx=y_idx,
                key=keys_mstep[it],
                mc_samples_train=mc_samples_train,
                lambda_y=lambda_y,
                gamma_mu_hidden=gamma_mu_hidden,
                gamma_mu_output=gamma_mu_output,
            )
            if first_m_diag is None:
                first_m_diag = m_diag
        # Compose per-layer first/last diagnostics across the m_step_iters inner loop.
        m_diag = {
            k: _with_loop_kl_summary(first_m_diag[k], m_diag[k])
            for k in m_diag
        }

        # F_DPC decomposition snapshot AFTER M-step.
        f_dpc_terms = shared_energy_terms(
            new_net, x, y_mean, y_var, frozen.m_zs, frozen.v_zs,
            y_idx=y_idx, key=key_snapshot, mc_samples_train=mc_samples_train,
            gamma_hidden=gamma_hidden, gamma_output=gamma_output,
            gamma_mu_hidden=gamma_mu_hidden,
            gamma_mu_output=gamma_mu_output,
            weight_kl_scale=weight_kl_scale,
            output_weight=lambda_y,
        )
        return new_net, e_diag, m_diag, f_dpc_terms, init_residuals, freeze_residuals

    return batch_step
