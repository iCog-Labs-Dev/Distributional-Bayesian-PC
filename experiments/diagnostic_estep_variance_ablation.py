"""Compare fixed, capped, and no-hidden-log-ratio E-step variants."""

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from bpcn.data.types import Targets
from bpcn.inference.shared_energy import latent_free_energy, output_energy
from bpcn.inference.state import LatentState
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.models.moments import moment_forward
from bpcn.training.m_step import m_step
from bpcn.utils.safe_math import (
    MIN_LATENT_LOG_VARIANCE, clamp_latent_log_variance, floor_variance,
)


def make_estep(mode, value=None):
    def infer(network, inputs, targets, config, key):
        network = jax.lax.stop_gradient(network)
        cfg = config.inference
        predictive = moment_forward(network.layers[0], inputs, jnp.zeros_like(inputs))
        mean = predictive.mean
        logvar = jnp.log(jnp.maximum(predictive.variance, cfg.initial_variance))
        if mode == "fixed":
            logvar = jnp.full_like(mean, jnp.log(value))
        elif mode == "capped":
            logvar = jnp.clip(logvar, MIN_LATENT_LOG_VARIANCE, jnp.log(value))

        def objective(mean, logvar):
            state = LatentState(means=(mean,), variances=(jnp.exp(logvar),))
            if mode != "no_log_ratio":
                return latent_free_energy(
                    network, inputs, state, targets=targets, key=key,
                    mc_samples=config.update.training_mc_samples,
                    output_loss_weight=config.update.output_loss_weight,
                )
            # Same hidden quadratic/variance terms, with only the log ratio removed.
            hidden = 0.5 * (
                (state.variances[0] + (mean - predictive.mean)**2)
                / floor_variance(predictive.variance) - 1.0
            )
            return hidden.sum(axis=-1).mean() + config.update.output_loss_weight * output_energy(
                network, state, targets, key, config.update.training_mc_samples
            )

        gradients = jax.grad(objective, argnums=(0, 1))
        for _ in range(cfg.steps):
            dmean, dlogvar = gradients(mean, logvar)
            mean = mean - cfg.mean_learning_rate * dmean
            if mode != "fixed":
                logvar = logvar - cfg.log_variance_learning_rate * dlogvar
                if mode == "capped":
                    logvar = jnp.clip(logvar, MIN_LATENT_LOG_VARIANCE, jnp.log(value))
                else:
                    logvar = clamp_latent_log_variance(logvar)
        return jax.lax.stop_gradient(
            LatentState(means=(mean,), variances=(jnp.exp(logvar),))
        )
    return infer


def make_step(infer, config, protocol):
    @jax.jit
    def step(network, inputs, target_mean, target_variance, class_indices, key):
        ekey, mkey = jax.random.split(key)
        targets = Targets(
            mean=target_mean, variance=target_variance, class_indices=class_indices
        )
        latents = infer(network, inputs, targets, config, ekey)
        for subkey in jax.random.split(mkey, config.update.steps):
            network = m_step(
                network, latents, inputs, targets, config.update,
                data_scale=1.0 / inputs.shape[0],
                prior_scale=1.0 / protocol.training_samples, key=subkey,
            ).network
        return network
    return step


def train(infer, config, spec, batches, protocol):
    network, frozen_index = lg._build_network(config, spec, protocol)
    frozen = network.layers[frozen_index]
    frozen_mean = np.asarray(frozen.mean).copy()
    frozen_logvar = np.asarray(frozen.log_variance).copy()
    learned = lg._learned_index(spec.stage)
    step = make_step(infer, config, protocol)
    key = jax.random.PRNGKey(protocol.training_seed)
    stable = 0
    for epoch in range(1, protocol.max_epochs + 1):
        old_mean = np.asarray(network.layers[learned].mean, dtype=np.float64).copy()
        old_logvar = np.asarray(network.layers[learned].log_variance, dtype=np.float64).copy()
        for batch in batches:
            key, batch_key = jax.random.split(key)
            network = step(
                network, batch.inputs, batch.targets.mean, batch.targets.variance,
                batch.targets.class_indices, batch_key,
            )
        layer = network.layers[learned]
        mean = np.asarray(layer.mean, dtype=np.float64)
        logvar = np.asarray(layer.log_variance, dtype=np.float64)
        if not (np.isfinite(mean).all() and np.isfinite(logvar).all()):
            return network, "NON_FINITE", epoch
        delta = max(np.abs(mean - old_mean).max(), np.abs(logvar - old_logvar).max())
        stable = stable + 1 if delta < protocol.convergence_tolerance else 0
        if epoch == 1 or epoch % 100 == 0:
            print(f"  epoch={epoch} delta={delta:.4g}", flush=True)
        if stable >= protocol.convergence_patience:
            break
    lg._assert_frozen_layer_unchanged(network, frozen_index, frozen_mean, frozen_logvar)
    status = "COMPLETE" if stable >= protocol.convergence_patience else "NOT_CONVERGED"
    return network, status, epoch


def run():
    print("Preparing E-step ablations (9 training runs)...", flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    config = lg._build_bpcn_config(spec, protocol)
    if config.inference.initial_mean_perturbation_std != 0:
        raise ValueError("This diagnostic requires initial_mean_perturbation_std=0")
    target_variance = lg._resolve_target_variance(spec, protocol)
    x, z, y, test_x, _, test_y, true_weight = dtl._make_dataset_with_latent(spec, protocol)
    dtl._verify_dataset_match(x, y, test_x, test_y, true_weight, protocol)
    exact_xy, _ = dtl._exact_references(x, z, y, true_weight, protocol)
    data = lg._SyntheticData(
        train_inputs=x, train_targets=y, test_inputs=test_x,
        test_targets=test_y, true_weight=true_weight,
    )
    batches = lg._batches(data, target_variance, protocol)
    selected = np.random.default_rng(protocol.training_seed).permutation(
        protocol.training_samples
    )[:protocol.batch_size]
    batch = batches[0]
    np.testing.assert_array_equal(np.asarray(batch.inputs), x[selected].astype(np.float32))
    variants = [("fixed", 0.01)] + [
        ("capped", cap) for cap in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 8.0)
    ] + [("no_log_ratio", None)]

    for mode, value in variants:
        print(f"\n{mode}, value={value}", flush=True)
        infer = make_estep(mode, value)
        network, status, epochs = train(infer, config, spec, batches, protocol)
        print(f"  {status}; epochs={epochs}", flush=True)
        if status == "NON_FINITE":
            continue
        state = infer(
            network, batch.inputs, batch.targets, config,
            jax.random.PRNGKey(protocol.training_seed),
        )
        predictive = moment_forward(network.layers[0], batch.inputs, jnp.zeros_like(batch.inputs))
        residual = gaussian_kl(
            state.means[0], state.variances[0], predictive.mean, predictive.variance
        ).variance_residual
        metrics = lg._cell_metrics(network, spec, exact_xy, data, protocol)
        print(f"  Latent variance: {float(state.variances[0].mean()):.6g}")
        print(f"  Predictive variance: {float(predictive.variance.mean()):.6g}")
        print(f"  Variance residual: {float(residual.mean()):.6g}")
        print(f"  Weight variance: {float(network.layers[0].weight_variance().mean()):.6g}")
        print(f"  Latent mean error: {lg._relative_error(np.asarray(state.means[0]), z[selected])}")
        print(f"  Weight mean error vs exact(x,y): {metrics.relative_mean_error}")
        print(f"  Weight variance error vs exact(x,y): {metrics.relative_variance_error}")
        print(f"  Weight variance ratio: {metrics.weight_variance_ratio}", flush=True)


if __name__ == "__main__":
    run()
