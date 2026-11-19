"""Sweep mean learning rates at beta=2, inspecting sampled checkpoints."""

from dataclasses import replace

import jax
import numpy as np

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_variance_ablation import make_step
from experiments.diagnostic_estep_soft_penalty import make_estep
from experiments.diagnostic_soft_penalty_trajectory import inspect

BETA = 2.0
LEARNING_RATES = (0.05, 0.04, 0.03, 0.02, 0.015, 0.01)
CHECKPOINTS = {5, 10, 20, 35, 60, 100, 150, 250, 400, 700, 1200, 2000}


def run():
    print('Preparing learning-rate trajectory sweep (6 training runs)...', flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    config = lg._build_bpcn_config(spec, protocol)
    base = config
    if config.inference.initial_mean_perturbation_std != 0:
        raise ValueError('This diagnostic requires initial_mean_perturbation_std=0')
    target_variance = lg._resolve_target_variance(spec, protocol)
    x, z, y, test_x, _, test_y, true_weight = dtl._make_dataset_with_latent(spec, protocol)
    dtl._verify_dataset_match(x, y, test_x, test_y, true_weight, protocol)
    exact, _ = dtl._exact_references(x, z, y, true_weight, protocol)
    data = lg._SyntheticData(train_inputs=x, train_targets=y, test_inputs=test_x, test_targets=test_y, true_weight=true_weight)
    batches = lg._batches(data, target_variance, protocol)
    selected = np.random.default_rng(protocol.training_seed).permutation(protocol.training_samples)[:protocol.batch_size]
    np.testing.assert_array_equal(np.asarray(batches[0].inputs), x[selected].astype(np.float32))
    for learning_rate in LEARNING_RATES:
        print(f'\nMean learning rate={learning_rate:g}; beta=2', flush=True)
        config = replace(base, update=replace(base.update, hidden_mean_learning_rate=learning_rate))
        infer = make_estep(BETA)
        step = make_step(infer, config, protocol)
        network, frozen_index = lg._build_network(config, spec, protocol)
        frozen_mean = np.asarray(network.layers[frozen_index].mean).copy()
        frozen_logvar = np.asarray(network.layers[frozen_index].log_variance).copy()
        learned = lg._learned_index(spec.stage)
        key = jax.random.PRNGKey(protocol.training_seed)
        stable = 0
        sampled_max = -np.inf
        final_radius = None
        status = 'NOT_CONVERGED'
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
                status = 'NON_FINITE'
                final_radius = None
                break
            delta = max(np.abs(mean - old_mean).max(), np.abs(logvar - old_logvar).max())
            stable = stable + 1 if delta < protocol.convergence_tolerance else 0
            finished = stable >= protocol.convergence_patience
            if epoch == 1 or epoch % 100 == 0:
                print(f'  Training epoch {epoch}; delta={delta:.5g}', flush=True)
            if epoch in CHECKPOINTS or finished or epoch == protocol.max_epochs:
                print(f'Checkpoint epoch={epoch}; delta={delta:.5g}', flush=True)
                final_radius = inspect(network, infer, batches[0], config, protocol, z[selected], exact, data, spec)
                sampled_max = max(sampled_max, final_radius)
            if finished:
                status = 'COMPLETE'
                break
        lg._assert_frozen_layer_unchanged(network, frozen_index, frozen_mean, frozen_logvar)
        print(f'Result: {status}; epochs={epoch}')
        print(f'  Largest sampled radius: {(sampled_max if np.isfinite(sampled_max) else None)}')
        print(f'  Final radius: {final_radius}')
        if status != 'NON_FINITE' and final_radius is not None:
            print('  Earlier sampled radius > 1, final radius < 1:', bool(sampled_max > 1 and final_radius < 1), flush=True)


if __name__ == "__main__":
    run()
