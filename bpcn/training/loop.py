"""Algorithm 3: BPCN training loop driver.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 6.5 Algorithm 3.
- Section 6.7 stopping / diagnostics.

References (extension: shared_energy_dbpcn_extension.pdf):
- Section 6.3 Algorithm 3: shared-energy training loop.
- Eq. 12: F_DPC, the canonical shared free energy descended each batch.
- Eq. 72: F_out + F_trans-DPC + F_weight-KL decomposition (logged per batch).

The per-batch step is jit-compiled with the hyperparameter constants closed
over so JAX can fuse the E-step and M-step into a single graph (they still
do NOT share gradients -- see plan section 6.5).
"""
from functools import partial
import jax

from ..inference.e_step import e_step
from ..inference.shared_energy import shared_energy_terms
from .m_step import m_step


def _with_loop_kl_summary(first_diag, final_diag):
    """Attach full inner-loop KL movement to the final-step diagnostics."""
    return final_diag._replace(
        kl_data_loop_before=first_diag.kl_data_before,
        kl_data_loop_after=final_diag.kl_data_after,
        kl_data_loop_delta=final_diag.kl_data_after - first_diag.kl_data_before,
    )


def make_batch_step(cfg, N_train: int):
    """Build a jit-compiled function batch_step(net, x, y_mean, y_var, y_idx, key) -> BatchOutcome.

    Uses the SGD-per-data-point scaling convention (see training/m_step.py
    docstring):
        data_scale  = 1/B
        prior_scale = 1/N_train
    so that eta values live in the conventional 1e-3 range. Mathematically
    equivalent to Eq. 81 with eta rescaled by 1/N_train.

    Returns `(new_net, e_diag, m_diag, f_dpc_terms)` per batch. `f_dpc_terms`
    is the post-M-step decomposition of F_DPC (extension Eq. 72).
    """
    data_scale = 1.0 / float(cfg.batch_size)
    prior_scale = 1.0 / float(N_train)
    # F_DPC diagnostic must use the same per-data-point scale as the M-step's
    # gradient. Data terms f_out/f_trans_dpc are already batch means (1/B);
    # the weight KL gets the matching 1/N_train multiplier so the assembled
    # F_DPC reflects the scalar the M-step descended. See
    # bpcn/inference/shared_energy.py weight_kl_scale docstring.
    weight_kl_scale = 1.0 / float(N_train)

    m_step_iters = int(cfg.m_step_iters)
    if m_step_iters < 1:
        raise ValueError(f"m_step_iters must be >= 1, got {m_step_iters}")

    # Capture cfg constants in the closure so JAX treats them as static.
    gamma_hidden = float(cfg.gamma_hidden)
    gamma_output = float(cfg.gamma_output)
    # Categorical-output extension constants (continuation note Eq. 2/48).
    mc_samples_train = int(cfg.mc_samples_train)
    lambda_y = float(cfg.lambda_y)

    @partial(jax.jit, static_argnums=())
    def batch_step(net, x, y_mean, y_var, y_idx, key):
        # Split the batch key into (E-step key, M-step keys, F_DPC-snapshot
        # key). M-step inner iters get fresh randomness independent of the
        # E-step's MC samples.
        n_keys = 2 + m_step_iters
        keys = jax.random.split(key, n_keys)
        key_estep = keys[0]
        keys_mstep = keys[1:1 + m_step_iters]
        key_snapshot = keys[-1]
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

        # F_DPC decomposition snapshot AFTER M-step (extension Eq. 72).
        # `output_weight=lambda_y` makes the returned f_out match the scalar
        # the M-step descended, so f_out + f_trans_dpc + f_weight_kl reads as
        # F_cat-DPC. Pass the *tuple* of per-layer latents (frozen.m_zs /
        # frozen.v_zs) so the multi-layer F_trans_dpc sum visits every hidden
        # layer. Under L_hidden==1 these are 1-tuples and the result matches
        # the legacy single-array call exactly.
        f_dpc_terms = shared_energy_terms(
            new_net, x, y_mean, y_var, frozen.m_zs, frozen.v_zs,
            y_idx=y_idx, key=key_snapshot, mc_samples_train=mc_samples_train,
            gamma_hidden=gamma_hidden, gamma_output=gamma_output,
            weight_kl_scale=weight_kl_scale,
            output_weight=lambda_y,
        )
        return new_net, e_diag, m_diag, f_dpc_terms

    return batch_step
