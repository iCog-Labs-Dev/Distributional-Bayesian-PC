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

from ..models.moments import moment_forward
from ..inference.e_step import e_step


def mc_predict(net, x, y_dummy, *, T_z, eta_m, eta_u, v_init, key, S: int):
    """Monte Carlo predictive (Eq. 103) using target-free test-time E-step.

    Per Section 6.6 paragraph 1: "Given x_*, clamp z_*^0 = x_* and run the
    E-step without a target if doing unsupervised or OOD scoring". We run
    e_step with output_weight=0.0 so the latent posterior is anchored only
    by the transition prior and the latent entropy.

    Returns: p_hat [B, C], probabilities averaged over S weight samples.
    """
    B, C = x.shape[0], net.layers[-1].d_out
    placeholder_target = jnp.zeros((B, C), dtype=x.dtype)
    frozen, _ = e_step(
        net, x, placeholder_target,
        T_z=T_z, eta_m=eta_m, eta_u=eta_u, v_init=v_init,
        output_weight=0.0,
    )
    m_z, v_z = frozen.m_z, frozen.v_z

    # MC-sample W_y from q(W_y) and z^1 from q^*(z^1), then softmax.
    output = net.layers[-1]
    sigma_y = jnp.sqrt(jnp.exp(output.tau))
    sigma_z = jnp.sqrt(v_z)

    def one_sample(k):
        k_W, k_z = jax.random.split(k)
        W_sample = output.mu + sigma_y * jax.random.normal(k_W, output.mu.shape)
        z_sample = m_z + sigma_z * jax.random.normal(k_z, m_z.shape)
        logits = z_sample @ W_sample.T                     # [B, C]
        return jax.nn.softmax(logits, axis=-1)

    keys = jax.random.split(key, S)
    probs = jax.vmap(one_sample)(keys)                     # [S, B, C]
    return probs.mean(axis=0)


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
    """Run MC predictive on a Split (data/mnist.py); returns dict of metrics."""
    N = len(split.x)
    n_classes = net.layers[-1].d_out
    correct = 0
    total = 0
    log_lik_sum = 0.0
    entropies = []
    keys = jax.random.split(key, (N + batch_size - 1) // batch_size + 1)
    ki = 0
    for i in range(0, N, batch_size):
        sl = slice(i, min(i + batch_size, N))
        x = jnp.asarray(split.x[sl])
        y_idx = jnp.asarray(split.y_idx[sl])
        ki += 1
        p_hat = mc_predict(
            net, x, None,
            T_z=cfg.eval_T_z_resolved,
            eta_m=cfg.eval_eta_m_resolved,
            eta_u=cfg.eval_eta_u_resolved,
            v_init=cfg.eval_v_init_resolved,
            key=keys[ki], S=cfg.mc_samples,
        )
        pred = jnp.argmax(p_hat, axis=-1)
        correct += int(jnp.sum(pred == y_idx))
        total += int(y_idx.shape[0])
        ll = jnp.log(jnp.clip(p_hat[jnp.arange(y_idx.shape[0]), y_idx], 1e-12, 1.0))
        log_lik_sum += float(ll.sum())
        entropies.append(np.asarray(predictive_entropy(p_hat)))
    entropies = np.concatenate(entropies)
    return {
        "accuracy": correct / max(total, 1),
        "log_likelihood_mean": log_lik_sum / max(total, 1),
        "entropy_mean": float(entropies.mean()),
        "entropy_std": float(entropies.std()),
        "n": total,
    }
