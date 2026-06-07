"""Test-time prediction and uncertainty extraction.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 6.6.
- Eq. 102: predictive distribution = integral over q(z) q(W).
- Eq. 103: MC classification predictive (1/S) sum softmax(W_y_s z^L_s).
- Eq. 104: predictive entropy H[y | x*, D].
- Eq. 105: epistemic proxy = mutual information.

For BPCN at test time, we run the target-free E-step on the test input
(Section 6.6 first paragraph), then MC-sample from q(z^L) and q(W_y) for
the categorical predictive distribution in Eq. 103.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..inference.e_step import e_step
from ..inference.feature_moments import psi_moments, apply_psi_sample


def _target_free_frozen(
    net, x, *,
    T_z, eta_m, eta_u, v_init,
    y_var=None,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
):
    """Run the target-free test-time E-step (v2 Section 6.6 paragraph 1).

    Descends F_DPC (extension Eq. 12) with `output_weight=0.0` so the latent
    is anchored only by the hidden transition. Returns the frozen (m_z, v_z)
    posterior used by `evaluate_split` (via `_mc_predict_from_frozen` and
    `_mean_predict_from_frozen`) so MC and MEAN predictives share one
    target-free E-step.
    """
    B = x.shape[0]
    C = net.layers[-1].d_out
    placeholder_target = jnp.zeros((B, C), dtype=x.dtype)
    if y_var is None:
        y_var = jnp.zeros_like(placeholder_target)
    frozen, _ = e_step(
        net, x, placeholder_target,
        T_z=T_z, eta_m=eta_m, eta_u=eta_u, v_init=v_init,
        output_weight=0.0,
        y_var=y_var,
        gamma_hidden=gamma_hidden,
        gamma_output=gamma_output,
    )
    return frozen


def _mean_predict_from_frozen(net, frozen):
    # Presynaptic feature to the output layer is psi_1(m_z) (Section 4.5).
    m_h, _ = psi_moments(net.activations[-1], frozen.m_z, frozen.v_z)
    return jax.nn.softmax(m_h @ net.layers[-1].mu.T, axis=-1)


def _mc_predict_from_frozen(net, frozen, key, S: int):
    output = net.layers[-1]
    sigma_y = jnp.sqrt(jnp.exp(output.tau))
    sigma_z = jnp.sqrt(frozen.v_z)
    m_z = frozen.m_z
    # Top-latent activation psi_L is the only one consumed by the output
    # head (continuation Eq. 26: a^(s) = W^(s) psi_L(z^(L,(s)))).
    psi_top = net.activations[-1]

    def one_sample(k):
        k_W, k_z = jax.random.split(k)
        W_sample = output.mu + sigma_y * jax.random.normal(k_W, output.mu.shape)
        z_sample = m_z + sigma_z * jax.random.normal(k_z, m_z.shape)
        # Apply psi_L to the sample directly (Section 4.5 option 4: MC samples
        # through psi). This is exact for the sample; the delta-method
        # approximation used by `psi_moments` is only needed when no samples
        # are available.
        z_sample = apply_psi_sample(psi_top, z_sample)
        logits = z_sample @ W_sample.T                     # [B, C]
        return jax.nn.softmax(logits, axis=-1)

    keys = jax.random.split(key, S)
    probs = jax.vmap(one_sample)(keys)                     # [S, B, C]
    return probs.mean(axis=0)


def predictive_entropy(p_hat):
    """H[y | x*, D] = -sum_c p_c log p_c  (Eq. 104)."""
    p = jnp.clip(p_hat, 1e-12, 1.0)
    return -jnp.sum(p * jnp.log(p), axis=-1)


def evaluate_split(net, split, cfg, key, *, batch_size: int = 256):
    """Run MC and posterior-mean predictives on a Split; returns merged metrics.

    Returns the standard MC keys (`accuracy`, `log_likelihood_mean`,
    `entropy_mean`, `entropy_std`, `n`) plus a parallel `mean_*` set computed
    from `_mean_predict_from_frozen` over the same target-free E-step latents.
    Both passes share the same per-batch RNG split for MC; the mean pass uses
    no RNG.
    """
    N = len(split.x)
    correct_mc = correct_mean = 0
    total = 0
    log_lik_mc = log_lik_mean = 0.0
    entropies_mc, entropies_mean = [], []
    keys = jax.random.split(key, (N + batch_size - 1) // batch_size + 1)
    ki = 0
    for i in range(0, N, batch_size):
        sl = slice(i, min(i + batch_size, N))
        x = jnp.asarray(split.x[sl])
        y_idx = jnp.asarray(split.y_idx[sl])
        ki += 1
        # Share one target-free E-step between both predictives so the only
        # difference is sampling-vs-mean, not which latent posterior was used.
        # The eval E-step descends the same shared energy F_DPC (extension
        # Eq. 12) as training.
        frozen = _target_free_frozen(
            net, x,
            T_z=cfg.eval_T_z_resolved,
            eta_m=cfg.eval_eta_m_resolved,
            eta_u=cfg.eval_eta_u_resolved,
            v_init=cfg.eval_v_init_resolved,
            gamma_hidden=cfg.gamma_hidden,
            gamma_output=cfg.gamma_output,
        )
        p_mc = _mc_predict_from_frozen(net, frozen, keys[ki], cfg.mc_samples)
        p_mean = _mean_predict_from_frozen(net, frozen)
        idx = jnp.arange(y_idx.shape[0])
        correct_mc   += int(jnp.sum(jnp.argmax(p_mc,   axis=-1) == y_idx))
        correct_mean += int(jnp.sum(jnp.argmax(p_mean, axis=-1) == y_idx))
        total += int(y_idx.shape[0])
        log_lik_mc   += float(jnp.log(jnp.clip(p_mc  [idx, y_idx], 1e-12, 1.0)).sum())
        log_lik_mean += float(jnp.log(jnp.clip(p_mean[idx, y_idx], 1e-12, 1.0)).sum())
        entropies_mc  .append(np.asarray(predictive_entropy(p_mc)))
        entropies_mean.append(np.asarray(predictive_entropy(p_mean)))
    entropies_mc   = np.concatenate(entropies_mc)
    entropies_mean = np.concatenate(entropies_mean)
    denom = max(total, 1)
    return {
        "accuracy": correct_mc / denom,
        "log_likelihood_mean": log_lik_mc / denom,
        "entropy_mean": float(entropies_mc.mean()),
        "entropy_std":  float(entropies_mc.std()),
        "mean_accuracy": correct_mean / denom,
        "mean_log_likelihood_mean": log_lik_mean / denom,
        "mean_entropy_mean": float(entropies_mean.mean()),
        "mean_entropy_std":  float(entropies_mean.std()),
        "n": total,
    }
