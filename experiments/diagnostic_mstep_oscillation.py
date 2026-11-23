"""Inspect M-step variance oscillation using the true-latent setup."""

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from bpcn.data.types import Targets
from bpcn.inference.state import LatentState
from bpcn.training.gradients import compute_layer_gradients
from bpcn.training.m_step import m_step
from bpcn.utils.safe_math import clamp_weight_log_variance

CHECKPOINTS = (50, 200, 500, 1000, 1500, 2000)
TRACE_EPOCHS = 3


def fine_trace(network, config, batches, protocol, key):
    @jax.jit
    def one_update(network, inputs, z, y, y_variance, step_key):
        latents = LatentState(
            means=(z,),
            variances=(jnp.full_like(z, dtl._TRUE_Z_VARIANCE),),
        )
        targets = Targets(mean=y, variance=y_variance, class_indices=None)
        return m_step(
            network, latents, inputs, targets, config.update,
            data_scale=1.0 / inputs.shape[0],
            prior_scale=1.0 / protocol.training_samples,
            key=step_key,
        ).network

    trace = []
    for _ in range(TRACE_EPOCHS):
        for batch in batches:
            key, batch_key = jax.random.split(key)
            for step_key in jax.random.split(batch_key, config.update.steps):
                network = one_update(
                    network, batch["inputs"], batch["true_z"],
                    batch["target_mean"], batch["target_variance"], step_key,
                )
                variance = np.asarray(network.layers[0].weight_variance())
                if not np.isfinite(variance).all():
                    raise FloatingPointError("Non-finite variance during trace")
                trace.append((float(variance.mean()), float(variance[0, 0])))
    return np.asarray(trace)


def _stability_analysis(network, config, batches, protocol):
    layer = network.layers[0]
    batch = batches[0]
    data_scale = 1.0 / float(batch['inputs'].shape[0])
    prior_scale = 1.0 / float(protocol.training_samples)
    input_mean = batch['inputs']
    input_variance = jnp.zeros_like(input_mean)
    target_mean = batch['true_z']
    target_variance = jnp.full_like(target_mean, dtl._TRUE_Z_VARIANCE)
    learning_rate = config.update.hidden_log_variance_learning_rate
    weight_kl_scale = config.update.hidden_weight_kl_scale
    mean_kl_scale = config.update.hidden_mean_kl_scale

    def row_update_map(row, output_index):
        full_log_variance = layer.log_variance.at[output_index, :].set(row)
        candidate = layer._replace(log_variance=full_log_variance)
        gradients = compute_layer_gradients(candidate, input_mean, input_variance, target_mean, target_variance, data_scale=data_scale, prior_scale=prior_scale, weight_kl_scale=weight_kl_scale, mean_kl_scale=mean_kl_scale)
        updated = clamp_weight_log_variance(candidate.log_variance + learning_rate * gradients.log_variance_gradient)
        return updated[output_index, :]
    per_row = []
    for output_index in range(layer.output_dim):
        row = layer.log_variance[output_index, :]
        jacobian = jax.jacfwd(lambda r, oi=output_index: row_update_map(r, oi))(row)
        eigenvalues = np.linalg.eigvals(np.asarray(jacobian, dtype=np.float64))
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        per_row.append({'output_index': output_index, 'spectral_radius': spectral_radius, 'weight_variance_row_mean': float(np.asarray(layer.weight_variance()[output_index], dtype=np.float64).mean())})
    return np.asarray([row['spectral_radius'] for row in per_row])

def run():
    print("Preparing true-latent diagnostic...", flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    config = lg._build_bpcn_config(spec, protocol)
    target_variance = lg._resolve_target_variance(spec, protocol)
    x, z, y, test_x, _, test_y, true_weight = dtl._make_dataset_with_latent(
        spec, protocol
    )
    dtl._verify_dataset_match(x, y, test_x, test_y, true_weight, protocol)
    batches = dtl._bypass_batches(x, z, y, target_variance, protocol)
    network, _ = lg._build_network(config, spec, protocol)
    step = dtl._make_bypass_step(config, protocol)
    key = jax.random.PRNGKey(protocol.training_seed)

    for epoch in range(1, max(CHECKPOINTS) + 1):
        for batch in batches:
            key, batch_key = jax.random.split(key)
            network = step(
                network, batch["inputs"], batch["true_z"],
                batch["target_mean"], batch["target_variance"], batch_key,
            )

        layer = network.layers[0]
        if not (
            np.isfinite(np.asarray(layer.mean)).all()
            and np.isfinite(np.asarray(layer.log_variance)).all()
        ):
            raise FloatingPointError(f"Non-finite weights at epoch {epoch}")

        if epoch == 1 or epoch % 100 == 0:
            print(f"Training epoch {epoch}", flush=True)
        if epoch not in CHECKPOINTS:
            continue

        radii = _stability_analysis(network, config, batches, protocol)
        trace = fine_trace(network, config, batches, protocol, key)
        print(f"\nCheckpoint {epoch}", flush=True)
        print(f"  Variance mean: {float(layer.weight_variance().mean()):.6g}")
        print(f"  Row spectral radii: {np.array2string(radii, precision=4)}")
        print(f"  Rows with radius > 1: {int((radii > 1).sum())}/{len(radii)}")
        print("  First 12 updates: [mean variance, variance[0,0]]")
        print(trace[:12], flush=True)
        print(
            f"  Full trace ranges: mean={trace[:, 0].min():.6g}"
            f"..{trace[:, 0].max():.6g}, "
            f"[0,0]={trace[:, 1].min():.6g}..{trace[:, 1].max():.6g}",
            flush=True,
        )


if __name__ == "__main__":
    run()
