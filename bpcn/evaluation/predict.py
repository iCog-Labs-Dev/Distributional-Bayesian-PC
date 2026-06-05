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
    objective: str = "pc_free_energy",
    kappa: float = 1.0,
    y_var=None,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
):
    """Run the target-free test-time E-step (Section 6.6 paragraph 1).

    Returns the frozen (m_z, v_z) latent posterior obtained by descending the
    chosen `objective` with `output_weight=0.0` (no target). Shared by
    `mc_predict`, `mean_predict`, and `evaluate_split` so the same frozen
    latents are used across diagnostics.

    `objective`, `kappa`, `y_var`, `gamma_hidden`, `gamma_output` are
    forwarded to `e_step` so the eval E-step can match the *training*
    objective (e.g. `objective="shared_dpc"` for shared-DPC-trained models),
    keeping train and test inference internally consistent.
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
        objective=objective,
        kappa=kappa,
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


def mean_predict(
    net, x, *,
    T_z, eta_m, eta_u, v_init,
    objective: str = "pc_free_energy",
    kappa: float = 1.0,
    y_var=None,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
):
    """Posterior-mean predictive: softmax(m_z @ mu_y.T), no MC sampling.

    Runs the same target-free test-time E-step as `mc_predict` (Section 6.6
    paragraph 1) but skips MC sampling of W and z. The resulting prediction
    uses the posterior mean weights and the E-step's final latent mean
    directly. Pair with `mc_predict` to disentangle MC sampling noise from
    a genuinely diffuse weight posterior: if mean entropy << MC entropy at
    the same accuracy, the wide MC predictive is sampling variance, not
    posterior σ² that has anything to say.

    See `_target_free_frozen` for the `objective` / `kappa` / `y_var` /
    `gamma_*` kwargs: defaults reproduce legacy F_z target-free eval,
    `objective="shared_dpc"` matches shared-DPC training.
    """
    frozen = _target_free_frozen(
        net, x,
        T_z=T_z, eta_m=eta_m, eta_u=eta_u, v_init=v_init,
        objective=objective, kappa=kappa, y_var=y_var,
        gamma_hidden=gamma_hidden, gamma_output=gamma_output,
    )
    return _mean_predict_from_frozen(net, frozen)


def mc_predict(
    net, x, y_dummy, *,
    T_z, eta_m, eta_u, v_init, key, S: int,
    objective: str = "pc_free_energy",
    kappa: float = 1.0,
    y_var=None,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
):
    """Monte Carlo predictive (Eq. 103) using target-free test-time E-step.

    Per Section 6.6 paragraph 1: "Given x_*, clamp z_*^0 = x_* and run the
    E-step without a target if doing unsupervised or OOD scoring". We run
    e_step with output_weight=0.0 so the latent posterior is anchored only
    by the transition prior and the latent entropy.

    See `_target_free_frozen` for the `objective` / `kappa` / `y_var` /
    `gamma_*` kwargs: defaults reproduce legacy F_z target-free eval,
    `objective="shared_dpc"` matches shared-DPC training.

    Returns: p_hat [B, C], probabilities averaged over S weight samples.
    """
    frozen = _target_free_frozen(
        net, x,
        T_z=T_z, eta_m=eta_m, eta_u=eta_u, v_init=v_init,
        objective=objective, kappa=kappa, y_var=y_var,
        gamma_hidden=gamma_hidden, gamma_output=gamma_output,
    )
    return _mc_predict_from_frozen(net, frozen, key, S)


def predictive_entropy(p_hat):
    """H[y | x*, D] = -sum_c p_c log p_c  (Eq. 104)."""
    p = jnp.clip(p_hat, 1e-12, 1.0)
    return -jnp.sum(p * jnp.log(p), axis=-1)


def accuracy(p_hat, y_true_idx):
    """Argmax accuracy."""
    pred = jnp.argmax(p_hat, axis=-1)
    return jnp.mean(pred == y_true_idx)


def predictive_log_likelihood(p_hat, y_true_idx):
    """Mean log p_hat(y_true | x)."""
    p = jnp.clip(p_hat, 1e-12, 1.0)
    return jnp.mean(jnp.log(p[jnp.arange(p.shape[0]), y_true_idx]))


def evaluate_split(net, split, cfg, key, *, batch_size: int = 256):
    """Run MC and posterior-mean predictives on a Split; returns merged metrics.

    Returns the standard MC keys (`accuracy`, `log_likelihood_mean`,
    `entropy_mean`, `entropy_std`, `n`) plus a parallel `mean_*` set computed
    from `mean_predict` over the same target-free E-step latents. Both passes
    share the same per-batch RNG split for MC; the mean pass uses no RNG.
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
        # Eval objective defaults to the training objective via cfg, keeping
        # the test-time E-step consistent with the scalar the model was
        # trained against. Override via cfg.eval_objective if needed.
        frozen = _target_free_frozen(
            net, x,
            T_z=cfg.eval_T_z_resolved,
            eta_m=cfg.eval_eta_m_resolved,
            eta_u=cfg.eval_eta_u_resolved,
            v_init=cfg.eval_v_init_resolved,
            objective=cfg.eval_objective_resolved,
            kappa=1.0,
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
