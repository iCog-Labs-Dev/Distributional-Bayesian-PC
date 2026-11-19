"""Compare uncertainty scaling with dataset size under two prior-weight rules."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import linregress

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_soft_penalty import make_estep
from bpcn.data.types import Targets
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.models.moments import moment_forward
from bpcn.training.m_step import m_step

_SPEC = dtl._SPEC
N_VALUES = (124, 252, 500, 1000, 2000, 4000)
SEEDS = (0, 1, 2)


def _make_batch_step(e_step_fn, config, update_steps: int, prior_scale: float):

    @jax.jit
    def batch_step(network, inputs, target_mean, target_variance, class_indices, key):
        data_scale = 1.0 / float(inputs.shape[0])
        e_key, m_key = jax.random.split(key)
        targets = Targets(mean=target_mean, variance=target_variance, class_indices=class_indices)
        latents = e_step_fn(network, inputs, targets, config, e_key)
        keys = jax.random.split(m_key, update_steps)
        updated_network = network
        for index in range(update_steps):
            result = m_step(updated_network, latents, inputs, targets, config.update, data_scale=data_scale, prior_scale=prior_scale, key=keys[index])
            updated_network = result.network
        return updated_network
    return batch_step

def _measure(network, config, protocol, batches, e_step_fn, exact_xy, synthetic_data, prior_scale: float):
    batch0 = batches[0]
    key = jax.random.PRNGKey(protocol.training_seed)
    latents = e_step_fn(network, batch0.inputs, batch0.targets, config, key)
    m_z, v_z = (latents.means[0], latents.variances[0])
    predictive = moment_forward(network.layers[0], batch0.inputs, jnp.zeros_like(batch0.inputs))
    terms = gaussian_kl(m_z, v_z, predictive.mean, predictive.variance)
    v_p = float(np.asarray(predictive.variance).mean())
    v_z_mean = float(np.asarray(v_z).mean())
    r = float(np.asarray(terms.variance_residual).mean())
    e_mean_sq = float((np.asarray(terms.mean_error) ** 2).mean())
    weight_variance = float(np.asarray(network.layers[0].weight_variance(), dtype=np.float64).mean())
    metrics = lg._cell_metrics(network, _SPEC, exact_xy, synthetic_data, protocol)
    data_scale = 1.0 / float(batch0.inputs.shape[0])
    from bpcn.training.gradients import compute_layer_gradients
    gradients = compute_layer_gradients(network.layers[0], batch0.inputs, jnp.zeros_like(batch0.inputs), m_z, v_z, data_scale=data_scale, prior_scale=prior_scale, weight_kl_scale=config.update.hidden_weight_kl_scale, mean_kl_scale=config.update.hidden_mean_kl_scale)
    g_data = float(np.asarray(gradients.log_variance_data_gradient).mean())
    g_prior = float(np.asarray(gradients.log_variance_prior_gradient).mean())
    return {'weight_variance': weight_variance, 'exact_xy_diagonal_covariance_mean': float(exact_xy.diagonal_covariance.mean()), 'weight_variance_ratio': metrics.weight_variance_ratio, 'relative_mean_error': metrics.relative_mean_error, 'relative_variance_error': metrics.relative_variance_error, 'coverage_95': metrics.coverage_95, 'v_p': v_p, 'v_z': v_z_mean, 'kappa': v_z_mean / v_p, 'r': r, 'r_sign': 'positive' if r > 0 else 'negative' if r < 0 else 'zero', 'e_mean_sq': e_mean_sq, 'g_data': g_data, 'g_prior': g_prior}

def train(infer, config, spec, batches, protocol, prior_scale):
    network, frozen_index = lg._build_network(config, spec, protocol)
    frozen = network.layers[frozen_index]
    frozen_mean = np.asarray(frozen.mean).copy()
    frozen_logvar = np.asarray(frozen.log_variance).copy()
    learned = lg._learned_index(spec.stage)
    step = _make_batch_step(infer, config, config.update.steps, prior_scale)
    key = jax.random.PRNGKey(protocol.training_seed)
    stable = 0
    for epoch in range(1, protocol.max_epochs + 1):
        old_mean = np.asarray(network.layers[learned].mean, dtype=np.float64).copy()
        old_logvar = np.asarray(network.layers[learned].log_variance, dtype=np.float64).copy()
        for batch in batches:
            key, batch_key = jax.random.split(key)
            network = step(network, batch.inputs, batch.targets.mean, batch.targets.variance, batch.targets.class_indices, batch_key)
        layer = network.layers[learned]
        mean = np.asarray(layer.mean, dtype=np.float64)
        logvar = np.asarray(layer.log_variance, dtype=np.float64)
        if not (np.isfinite(mean).all() and np.isfinite(logvar).all()):
            return (network, 'NON_FINITE', epoch)
        delta = max(np.abs(mean - old_mean).max(), np.abs(logvar - old_logvar).max())
        stable = stable + 1 if delta < protocol.convergence_tolerance else 0
        if epoch == 1 or epoch % 100 == 0:
            print(f'  epoch={epoch} delta={delta:.4g}', flush=True)
        if stable >= protocol.convergence_patience:
            break
    lg._assert_frozen_layer_unchanged(network, frozen_index, frozen_mean, frozen_logvar)
    status = 'COMPLETE' if stable >= protocol.convergence_patience else 'NOT_CONVERGED'
    return (network, status, epoch)

def run():
    print("N-scaling test: 6 sizes x 3 seeds x 2 prior modes = 36 runs", flush=True)
    print("Requested 125 and 250 replaced by 124 and 252 for four equal batches.", flush=True)
    base = lg._Protocol()
    results = []
    for mode in ("tracking", "pinned"):
        for n in N_VALUES:
            for seed in SEEDS:
                print(f"\n{mode}: N={n}, seed offset={seed}", flush=True)
                protocol = replace(
                    base, training_samples=n, batch_size=n // 4,
                    data_seed=base.data_seed + seed,
                    network_seed=base.network_seed + seed,
                    training_seed=base.training_seed + seed,
                )
                config = lg._build_bpcn_config(_SPEC, protocol)
                if config.inference.initial_mean_perturbation_std != 0:
                    raise ValueError("This diagnostic requires initial_mean_perturbation_std=0")
                target_variance = lg._resolve_target_variance(_SPEC, protocol)
                data = lg._make_dataset(_SPEC, protocol)
                batches = lg._batches(data, target_variance, protocol)
                prior_scale = 1.0 / n if mode == "tracking" else 1.0 / 256
                infer = make_estep(2.0)
                network, status, epochs = train(infer, config, _SPEC, batches, protocol, prior_scale)
                updates = epochs * len(batches) * config.update.steps
                print(
                    f"  {status}; epochs={epochs}; M-step updates={updates}; "
                    f"prior_scale={prior_scale:.6g}", flush=True,
                )
                if status == "NON_FINITE":
                    continue
                covariance = lg._noise_covariance(_SPEC, data.true_weight, protocol)
                exact = lg._exact_posterior(
                    data.train_inputs, data.train_targets, covariance, protocol.prior_std
                )
                measured = _measure(network, config, protocol, batches, infer, exact, data, prior_scale)
                for name, value in measured.items():
                    print(f"  {name}: {value}", flush=True)
                results.append(dict(mode=mode, n=n, seed=seed, status=status, **measured))

    print("\nLog-log scaling fits; converged runs only", flush=True)
    for mode in ("tracking", "pinned"):
        subset = [p for p in results if p["mode"] == mode and p["status"] == "COMPLETE"]
        print(f"{mode}: {len(subset)} converged runs out of {len(N_VALUES) * len(SEEDS)}")
        for field in ("weight_variance", "exact_xy_diagonal_covariance_mean"):
            valid = [p for p in subset if np.isfinite(p[field]) and p[field] > 0]
            if len({p["n"] for p in valid}) < 2:
                print(f"  {field}: insufficient converged dataset sizes")
                continue
            fit = linregress(np.log([p["n"] for p in valid]), np.log([p[field] for p in valid]))
            print(
                f"  {field}: slope={fit.slope:.4f}, "
                f"SE={fit.stderr:.4f}, R²={fit.rvalue**2:.4f}", flush=True,
            )


if __name__ == "__main__":
    run()
