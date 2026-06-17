"""
Test-time prediction and uncertainty extraction.

For BPCN at test time, we run the target-free E-step on the test input, then MC-sample 
from q(z^L) and q(W_y) for the categorical predictive distribution.

"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bpcn.inference.e_step import e_step
from bpcn.inference.feature_moments import psi_moments, apply_psi_sample


def _target_free_frozen(
    net, x, *,
    T_z, eta_m, eta_u, v_init,
    y_var=None,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
):
    """
    Run the target-free test-time E-step.

    Descends F_DPC with `output_weight=0.0` so the latent
    is anchored only by the hidden transition. Returns the frozen (m_z, v_z)
    posterior consumed by the MC and MEAN predictive helpers.
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
        init_perturb_std=0.0,
    )
    return frozen


def _mean_predict_from_frozen(net, frozen):
    """ Predictive probabilities from the frozen posterior mean. """
    m_h, _ = psi_moments(net.activations[-1], frozen.m_z, frozen.v_z)
    return jax.nn.softmax(m_h @ net.layers[-1].mu.T, axis=-1)


def _mc_predict_from_frozen(net, frozen, key, S: int):
    output = net.layers[-1]
    sigma_y = jnp.sqrt(jnp.exp(output.tau))
    sigma_z = jnp.sqrt(frozen.v_z)
    m_z = frozen.m_z
    # Top-latent activation psi_L is the only one consumed by the output
    psi_top = net.activations[-1]

    def one_sample(k):
        k_W, k_z = jax.random.split(k)
        W_sample = output.mu + sigma_y * jax.random.normal(k_W, output.mu.shape)
        z_sample = m_z + sigma_z * jax.random.normal(k_z, m_z.shape)
        # Apply psi_L to the sample
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


@partial(jax.jit, static_argnames=("T_z", "mc_samples"))
def _eval_one_batch(
    net, x_batch, key, *,
    T_z, eta_m, eta_u, v_init,
    gamma_hidden, gamma_output, mc_samples,
):
    """
    One fused XLA pass: target-free E-step + MC predict + MEAN predict.

    Returns
    -------
    p_mc   : [B, C]   Monte Carlo predictive probabilities (Eq. 103).
    p_mean : [B, C]   posterior-mean predictive probabilities.
    """
    frozen = _target_free_frozen(
        net, x_batch,
        T_z=T_z, eta_m=eta_m, eta_u=eta_u, v_init=v_init,
        gamma_hidden=gamma_hidden, gamma_output=gamma_output,
    )
    p_mc = _mc_predict_from_frozen(net, frozen, key, mc_samples)
    p_mean = _mean_predict_from_frozen(net, frozen)
    return p_mc, p_mean


class _EvalScalars(NamedTuple):
    """JAX-scalar payload returned by `_evaluate_padded_jit`."""
    correct_mc: jax.Array
    correct_mean: jax.Array
    log_lik_mc: jax.Array
    log_lik_mean: jax.Array
    H_mc_mean: jax.Array
    H_mc_std: jax.Array
    H_mean_mean: jax.Array
    H_mean_std: jax.Array
    n_valid: jax.Array


@partial(jax.jit, static_argnames=(
    "n_batches", "batch_size", "T_z", "mc_samples",
))
def _evaluate_padded_jit(
    net, x_padded, y_padded, mask, keys, *,
    n_batches, batch_size,
    T_z, eta_m, eta_u, v_init,
    gamma_hidden, gamma_output, mc_samples,
) -> _EvalScalars:
    """Scan `_eval_one_batch` over `[n_batches, batch_size, ...]` then reduce.

    The padded examples are masked out before every reduction, so they
    contribute exactly zero to all metrics — the result is mathematically
    identical to evaluating only the valid examples.
    """
    # Reshape inputs to [n_batches, batch_size, ...].
    x_b = x_padded.reshape(n_batches, batch_size, -1)
    y_b = y_padded.reshape(n_batches, batch_size)
    m_b = mask.reshape(n_batches, batch_size)

    def step(_carry, args):
        k, xb = args
        p_mc, p_mean = _eval_one_batch(
            net, xb, k,
            T_z=T_z, eta_m=eta_m, eta_u=eta_u, v_init=v_init,
            gamma_hidden=gamma_hidden, gamma_output=gamma_output,
            mc_samples=mc_samples,
        )
        return _carry, (p_mc, p_mean)

    _, (p_mc_b, p_mean_b) = jax.lax.scan(step, None, (keys, x_b))
    # Flatten [n_batches, batch_size, C] -> [N_padded, C].
    p_mc = p_mc_b.reshape(n_batches * batch_size, -1)
    p_mean = p_mean_b.reshape(n_batches * batch_size, -1)
    y_flat = y_b.reshape(n_batches * batch_size)
    mask_flat = m_b.reshape(n_batches * batch_size).astype(p_mc.dtype)

    correct_mc_bool = (jnp.argmax(p_mc, axis=-1) == y_flat)
    correct_mean_bool = (jnp.argmax(p_mean, axis=-1) == y_flat)
    correct_mc = jnp.sum(correct_mc_bool.astype(mask_flat.dtype) * mask_flat)
    correct_mean = jnp.sum(correct_mean_bool.astype(mask_flat.dtype) * mask_flat)

    idx = jnp.arange(p_mc.shape[0])
    log_lik_mc = jnp.sum(
        jnp.log(jnp.clip(p_mc[idx, y_flat], 1e-12, 1.0)) * mask_flat
    )
    log_lik_mean = jnp.sum(
        jnp.log(jnp.clip(p_mean[idx, y_flat], 1e-12, 1.0)) * mask_flat
    )

    H_mc = predictive_entropy(p_mc) * mask_flat
    H_mean = predictive_entropy(p_mean) * mask_flat

    n_valid = jnp.sum(mask_flat)
    denom = jnp.maximum(n_valid, 1.0)
    H_mc_mean = jnp.sum(H_mc) / denom
    H_mean_mean = jnp.sum(H_mean) / denom
    H_mc_std = jnp.sqrt(
        jnp.sum(((H_mc - H_mc_mean) * mask_flat) ** 2) / denom
    )
    H_mean_std = jnp.sqrt(
        jnp.sum(((H_mean - H_mean_mean) * mask_flat) ** 2) / denom
    )

    return _EvalScalars(
        correct_mc=correct_mc,
        correct_mean=correct_mean,
        log_lik_mc=log_lik_mc,
        log_lik_mean=log_lik_mean,
        H_mc_mean=H_mc_mean,
        H_mc_std=H_mc_std,
        H_mean_mean=H_mean_mean,
        H_mean_std=H_mean_std,
        n_valid=n_valid,
    )


def evaluate_split(net, split, cfg, key, *, batch_size=None):
    """
    Run MC and posterior-mean predictives on a Split; return metrics dict.

    The whole eval (target-free E-step + MC predictive + MEAN predictive +
    metric reductions) runs inside a single jitted graph via `jax.lax.scan`
    over fixed-size batches.
    """
    if batch_size is None:
        batch_size = int(cfg.batch_size)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    N = int(len(split.x))
    n_batches = (N + batch_size - 1) // batch_size
    N_padded = n_batches * batch_size
    pad = N_padded - N

    x_arr = jnp.asarray(split.x)
    y_arr = jnp.asarray(split.y_idx)
    if pad > 0:
        x_padded = jnp.concatenate(
            [x_arr, jnp.zeros((pad, x_arr.shape[1]), dtype=x_arr.dtype)],
            axis=0,
        )
        y_padded = jnp.concatenate(
            [y_arr, jnp.zeros((pad,), dtype=y_arr.dtype)],
            axis=0,
        )
    else:
        x_padded, y_padded = x_arr, y_arr
    mask = jnp.arange(N_padded) < N
    keys = jax.random.split(key, n_batches + 1)[1:]

    scalars = _evaluate_padded_jit(
        net, x_padded, y_padded, mask, keys,
        n_batches=n_batches, batch_size=batch_size,
        T_z=int(cfg.eval_T_z_resolved),
        eta_m=float(cfg.eval_eta_m_resolved),
        eta_u=float(cfg.eval_eta_u_resolved),
        v_init=float(cfg.eval_v_init_resolved),
        gamma_hidden=float(cfg.gamma_hidden),
        gamma_output=float(cfg.gamma_output),
        mc_samples=int(cfg.mc_samples),
    )

    # Single device->host sync block at the end.
    N_f = float(N)
    return {
        "accuracy": float(scalars.correct_mc) / N_f,
        "log_likelihood_mean": float(scalars.log_lik_mc) / N_f,
        "entropy_mean": float(scalars.H_mc_mean),
        "entropy_std": float(scalars.H_mc_std),
        "mean_accuracy": float(scalars.correct_mean) / N_f,
        "mean_log_likelihood_mean": float(scalars.log_lik_mean) / N_f,
        "mean_entropy_mean": float(scalars.H_mean_mean),
        "mean_entropy_std": float(scalars.H_mean_std),
        "n": N,
    }
