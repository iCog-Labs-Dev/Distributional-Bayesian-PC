"""Inspect soft-penalty training at selected epochs, including the final epoch."""

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_variance_ablation import make_step
from experiments.diagnostic_estep_soft_penalty import make_estep, _joint_row_spectral_radii
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.models.moments import moment_forward

BETAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)
CHECKPOINTS = {5, 10, 20, 35, 60, 100, 150, 250, 400, 700, 1200, 2000}


def inspect(network, infer, batch, config, protocol, true_z, exact, data, spec):
    state = infer(
        network, batch.inputs, batch.targets, config,
        jax.random.PRNGKey(protocol.training_seed),
    )
    predictive = moment_forward(network.layers[0], batch.inputs, jnp.zeros_like(batch.inputs))
    residual = gaussian_kl(
        state.means[0], state.variances[0], predictive.mean, predictive.variance
    ).variance_residual
    metrics = lg._cell_metrics(network, spec, exact, data, protocol)
    radii = np.asarray(_joint_row_spectral_radii(
        network, batch.inputs, state.means[0], state.variances[0], config, protocol
    ))
    print(
        f"  Vz={float(state.variances[0].mean()):.5g} "
        f"Vp1={float(predictive.variance.mean()):.5g} "
        f"rV={float(residual.mean()):.5g} "
        f"weight_var={float(network.layers[0].weight_variance().mean()):.5g}"
    )
    print(f"  Latent mean error: {lg._relative_error(np.asarray(state.means[0]), true_z)}")
    print(
        f"  Weight errors vs exact(x,y): mean={metrics.relative_mean_error}, "
        f"variance={metrics.relative_variance_error}"
    )
    print(
        f"  Row radii: min={radii.min():.6f}, max={radii.max():.6f}; "
        f"rows > 1: {int((radii > 1).sum())}/{len(radii)}", flush=True,
    )
    return float(radii.max())


def run():
    print("Preparing trajectory sweep (11 training runs)...", flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    config = lg._build_bpcn_config(spec, protocol)
    if config.inference.initial_mean_perturbation_std != 0:
        raise ValueError("This diagnostic requires initial_mean_perturbation_std=0")
    target_variance = lg._resolve_target_variance(spec, protocol)
    x, z, y, test_x, _, test_y, true_weight = dtl._make_dataset_with_latent(spec, protocol)
    dtl._verify_dataset_match(x, y, test_x, test_y, true_weight, protocol)
    exact, _ = dtl._exact_references(x, z, y, true_weight, protocol)
    data = lg._SyntheticData(
        train_inputs=x, train_targets=y, test_inputs=test_x,
        test_targets=test_y, true_weight=true_weight,
    )
    batches = lg._batches(data, target_variance, protocol)
    selected = np.random.default_rng(protocol.training_seed).permutation(
        protocol.training_samples
    )[:protocol.batch_size]
    np.testing.assert_array_equal(np.asarray(batches[0].inputs), x[selected].astype(np.float32))

    for beta in BETAS:
        print(f"\nBeta={beta:g}", flush=True)
        infer = make_estep(beta)
        step = make_step(infer, config, protocol)
        network, frozen_index = lg._build_network(config, spec, protocol)
        frozen_mean = np.asarray(network.layers[frozen_index].mean).copy()
        frozen_logvar = np.asarray(network.layers[frozen_index].log_variance).copy()
        learned = lg._learned_index(spec.stage)
        key = jax.random.PRNGKey(protocol.training_seed)
        stable = 0
        sampled_max = -np.inf
        final_radius = None
        status = "NOT_CONVERGED"

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
                status = "NON_FINITE"
                final_radius = None
                break
            delta = max(np.abs(mean - old_mean).max(), np.abs(logvar - old_logvar).max())
            stable = stable + 1 if delta < protocol.convergence_tolerance else 0
            finished = stable >= protocol.convergence_patience
            if epoch == 1 or epoch % 100 == 0:
                print(f"  Training epoch {epoch}; delta={delta:.5g}", flush=True)
            if epoch in CHECKPOINTS or finished or epoch == protocol.max_epochs:
                print(f"Checkpoint epoch={epoch}; delta={delta:.5g}", flush=True)
                final_radius = inspect(
                    network, infer, batches[0], config, protocol,
                    z[selected], exact, data, spec,
                )
                sampled_max = max(sampled_max, final_radius)
            if finished:
                status = "COMPLETE"
                break

        lg._assert_frozen_layer_unchanged(network, frozen_index, frozen_mean, frozen_logvar)
        print(f"Result: {status}; epochs={epoch}")
        print(f"  Largest sampled radius: {sampled_max if np.isfinite(sampled_max) else None}")
        print(f"  Final radius: {final_radius}")
        if status != "NON_FINITE" and final_radius is not None:
            print(
                "  Earlier sampled radius > 1, final radius < 1:",
                bool(sampled_max > 1 and final_radius < 1), flush=True,
            )


if __name__ == "__main__":
    run()
