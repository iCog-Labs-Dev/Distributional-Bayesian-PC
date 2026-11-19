"""Compare inferred latents with true latents"""

import math

import jax
import jax.numpy as jnp
import numpy as np

import experiments.linear_gaussian as lg
from bpcn.data.types import Targets
from bpcn.inference.state import LatentState
from bpcn.training.m_step import m_step
from bpcn.utils.safe_math import MIN_VARIANCE

_SPEC = lg.CellSpec(
    stage="B_hidden",
    case="independent",
    output_residual_variance=0.01,
    target_variance_mode="matched",
)
_TRUE_Z_VARIANCE = MIN_VARIANCE


def _make_dataset_with_latent(spec: lg.CellSpec, protocol: lg._Protocol):
    rng = np.random.default_rng(protocol.data_seed)
    dimension = protocol.dimension
    true_weight = rng.standard_normal((dimension, dimension)) * protocol.prior_std
    hidden_variance = lg._hidden_residual_variance(spec.stage, protocol)

    def emit(inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample_count = len(inputs)
        hidden_noise = math.sqrt(hidden_variance) * rng.standard_normal((sample_count, dimension))
        output_noise = math.sqrt(spec.output_residual_variance) * rng.standard_normal((sample_count, dimension))
        if spec.stage == 'A_output':
            hidden = inputs + hidden_noise
            target = hidden @ true_weight.T + output_noise
        else:
            hidden = inputs @ true_weight.T + hidden_noise
            target = hidden + output_noise
        return (hidden, target)
    train_inputs = lg._make_inputs(spec.case, protocol.training_samples, rng, protocol)
    train_hidden, train_targets = emit(train_inputs)
    test_inputs = rng.standard_normal((protocol.test_samples, dimension))
    if spec.case == 'collinear':
        test_inputs[:, protocol.collinear_rank:] = 0.0
    test_hidden, test_targets = emit(test_inputs)
    return (train_inputs.astype(np.float64), train_hidden.astype(np.float64), train_targets.astype(np.float64), test_inputs.astype(np.float64), test_hidden.astype(np.float64), test_targets.astype(np.float64), true_weight.astype(np.float64))

def _verify_dataset_match(train_inputs, train_targets, test_inputs, test_targets, true_weight, protocol) -> None:
    reference = lg._make_dataset(_SPEC, protocol)
    checks = {'train_inputs': (train_inputs, reference.train_inputs), 'train_targets': (train_targets, reference.train_targets), 'test_inputs': (test_inputs, reference.test_inputs), 'test_targets': (test_targets, reference.test_targets), 'true_weight': (true_weight, reference.true_weight)}
    for name, (observed, expected) in checks.items():
        if not np.array_equal(observed, expected):
            raise RuntimeError(f'diagnostic dataset reproduction diverged from lg._make_dataset on {name!r}; bypass and normal runs would not be comparable')

def _exact_references(train_inputs, train_hidden, train_targets, true_weight, protocol) -> tuple[lg._ExactPosterior, lg._ExactPosterior]:
    dimension = protocol.dimension
    hidden_variance = lg._hidden_residual_variance(_SPEC.stage, protocol)
    noise_covariance_xy = lg._noise_covariance(_SPEC, true_weight, protocol)
    exact_xy = lg._exact_posterior(train_inputs, train_targets, noise_covariance_xy, protocol.prior_std)
    noise_covariance_xz = hidden_variance * np.eye(dimension)
    exact_xz = lg._exact_posterior(train_inputs, train_hidden, noise_covariance_xz, protocol.prior_std)
    return (exact_xy, exact_xz)

def _bypass_batches(train_inputs, train_hidden, train_targets, target_variance, protocol):
    order = np.random.default_rng(protocol.training_seed).permutation(protocol.training_samples)
    batches = []
    for start in range(0, protocol.training_samples, protocol.batch_size):
        selected = order[start:start + protocol.batch_size]
        batches.append({'inputs': jnp.asarray(train_inputs[selected], dtype=jnp.float32), 'true_z': jnp.asarray(train_hidden[selected], dtype=jnp.float32), 'target_mean': jnp.asarray(train_targets[selected], dtype=jnp.float32), 'target_variance': jnp.full((len(selected), protocol.dimension), target_variance, dtype=jnp.float32)})
    return tuple(batches)

def _make_bypass_step(config, protocol):
    update_steps = config.update.steps
    prior_scale = 1.0 / float(protocol.training_samples)

    @jax.jit
    def step(network, inputs, true_z, target_mean, target_variance, key):
        data_scale = 1.0 / float(inputs.shape[0])
        latents = LatentState(means=(true_z,), variances=(jnp.full_like(true_z, _TRUE_Z_VARIANCE),))
        targets = Targets(mean=target_mean, variance=target_variance, class_indices=None)
        keys = jax.random.split(key, update_steps)
        updated_network = network
        for index in range(update_steps):
            result = m_step(updated_network, latents, inputs, targets, config.update, data_scale=data_scale, prior_scale=prior_scale, key=keys[index])
            updated_network = result.network
        return updated_network
    return step


def train_bypass(network, config, batches, protocol):
    step = _make_bypass_step(config, protocol)
    key = jax.random.PRNGKey(protocol.training_seed)
    stable_epochs = 0
    frozen_mean = np.asarray(network.layers[1].mean).copy()
    frozen_log_variance = np.asarray(network.layers[1].log_variance).copy()
    for epoch in range(1, protocol.max_epochs + 1):
        old_mean = np.asarray(network.layers[0].mean).copy()
        old_log_variance = np.asarray(
            network.layers[0].log_variance
        ).copy()

        for batch in batches:
            key, batch_key = jax.random.split(key)
            network = step(
                network,
                batch["inputs"],
                batch["true_z"],
                batch["target_mean"],
                batch["target_variance"],
                batch_key,
            )

        mean = np.asarray(network.layers[0].mean)
        log_variance = np.asarray(network.layers[0].log_variance)

        if not (
            np.isfinite(mean).all()
            and np.isfinite(log_variance).all()
        ):
            return network, "NON_FINITE", epoch

        delta = max(
            np.max(np.abs(mean - old_mean)),
            np.max(np.abs(log_variance - old_log_variance)),
        )
        stable_epochs = (
            stable_epochs + 1
            if delta < protocol.convergence_tolerance
            else 0
        )

        if epoch == 1 or epoch % 100 == 0:
            print(f"BYPASS epoch {epoch}: delta={delta:.3g}", flush=True)

        if stable_epochs >= protocol.convergence_patience:
            break

    lg._assert_frozen_layer_unchanged(
        network, 1, frozen_mean, frozen_log_variance
    )
    status = (
        "COMPLETE"
        if stable_epochs >= protocol.convergence_patience
        else "NOT_CONVERGED"
    )
    return network, status, epoch


def run():
    print("Preparing data...", flush=True)
    protocol = lg._Protocol()
    config = lg._build_bpcn_config(_SPEC, protocol)
    target_variance = lg._resolve_target_variance(_SPEC, protocol)

    x, z, y, test_x, _, test_y, true_weight = (
        _make_dataset_with_latent(_SPEC, protocol)
    )
    _verify_dataset_match(
        x, y, test_x, test_y, true_weight, protocol
    )
    data = lg._SyntheticData(
        train_inputs=x,
        train_targets=y,
        test_inputs=test_x,
        test_targets=test_y,
        true_weight=true_weight,
    )
    exact_xy, exact_xz = _exact_references(
        x, z, y, true_weight, protocol
    )

    print("Starting NORMAL training...", flush=True)
    network, _ = lg._build_network(config, _SPEC, protocol)
    normal = lg._train_cell(
        lg.make_batch_step(config, protocol.training_samples),
        network,
        _SPEC,
        data,
        target_variance,
        protocol,
    )
    print(
        f"NORMAL finished: {normal.status}, {normal.epochs} epochs",
        flush=True,
    )

    print("Starting BYPASS training...", flush=True)
    network, _ = lg._build_network(config, _SPEC, protocol)
    bypass_network, bypass_status, bypass_epochs = train_bypass(
        network,
        config,
        _bypass_batches(x, z, y, target_variance, protocol),
        protocol,
    )

    for name, network, reference, status, epochs in (
        ("NORMAL", normal.network, exact_xy, normal.status, normal.epochs),
        ("BYPASS", bypass_network, exact_xz, bypass_status, bypass_epochs),
    ):
        print(f"\n{name}: {status}, epochs={epochs}", flush=True)
        if status == "NON_FINITE":
            continue

        metrics = lg._cell_metrics(
            network, _SPEC, reference, data, protocol
        )
        print(f"  Mean error:     {metrics.relative_mean_error}")
        print(f"  Variance error: {metrics.relative_variance_error}")


if __name__ == "__main__":
    run()