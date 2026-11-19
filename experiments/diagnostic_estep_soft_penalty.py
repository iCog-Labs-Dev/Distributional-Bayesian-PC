"""Sweep a soft latent-variance penalty; inspect final M-step sensitivity."""

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_variance_ablation import train
from bpcn.inference.shared_energy import latent_free_energy
from bpcn.inference.state import LatentState
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.models.moments import moment_forward
from bpcn.training.gradients import compute_layer_gradients
from bpcn.utils.safe_math import clamp_latent_log_variance, clamp_weight_log_variance

BETAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
REFERENCE_VARIANCE = 0.01
STABILITY_THRESHOLD = 0.05


def make_estep(beta):
    def infer(network, inputs, targets, config, key):
        network = jax.lax.stop_gradient(network)
        cfg = config.inference
        predictive = moment_forward(network.layers[0], inputs, jnp.zeros_like(inputs))
        mean = predictive.mean
        logvar = jnp.log(jnp.maximum(predictive.variance, cfg.initial_variance))

        def objective(mean, logvar):
            state = LatentState(means=(mean,), variances=(jnp.exp(logvar),))
            base = latent_free_energy(
                network, inputs, state, targets=targets, key=key,
                mc_samples=config.update.training_mc_samples,
                output_loss_weight=config.update.output_loss_weight,
            )
            if beta == 0.0:
                return base
            log_ratio = logvar - jnp.log(REFERENCE_VARIANCE)
            penalty = (jnp.exp(log_ratio) - log_ratio - 1.0).sum(axis=-1).mean()
            return base + beta * penalty

        gradients = jax.grad(objective, argnums=(0, 1))

        def update(carry, _):
            mean, logvar = carry
            dmean, dlogvar = gradients(mean, logvar)
            return (
                mean - cfg.mean_learning_rate * dmean,
                clamp_latent_log_variance(logvar - cfg.log_variance_learning_rate * dlogvar),
            ), None

        (mean, logvar), _ = jax.lax.scan(update, (mean, logvar), None, length=cfg.steps)
        return jax.lax.stop_gradient(
            LatentState(means=(mean,), variances=(jnp.exp(logvar),))
        )
    return infer


def _joint_row_spectral_radii(network, input_mean, target_mean, target_variance, config, protocol):
    layer = network.layers[0]
    data_scale = 1.0 / float(input_mean.shape[0])
    prior_scale = 1.0 / float(protocol.training_samples)
    input_variance = jnp.zeros_like(input_mean)
    mean_learning_rate = config.update.hidden_mean_learning_rate
    log_variance_learning_rate = config.update.hidden_log_variance_learning_rate
    weight_kl_scale = config.update.hidden_weight_kl_scale
    mean_kl_scale = config.update.hidden_mean_kl_scale

    def joint_row_map(params, output_index):
        input_dim = layer.input_dim
        mean_row = params[:input_dim]
        log_variance_row = params[input_dim:]
        full_mean = layer.mean.at[output_index, :].set(mean_row)
        full_log_variance = layer.log_variance.at[output_index, :].set(log_variance_row)
        candidate = layer._replace(mean=full_mean, log_variance=full_log_variance)
        gradients = compute_layer_gradients(candidate, input_mean, input_variance, target_mean, target_variance, data_scale=data_scale, prior_scale=prior_scale, weight_kl_scale=weight_kl_scale, mean_kl_scale=mean_kl_scale)
        new_mean = full_mean + mean_learning_rate * gradients.mean_gradient
        new_log_variance = clamp_weight_log_variance(full_log_variance + log_variance_learning_rate * gradients.log_variance_gradient)
        return jnp.concatenate([new_mean[output_index, :], new_log_variance[output_index, :]])
    radii = []
    for output_index in range(layer.output_dim):
        params0 = jnp.concatenate([layer.mean[output_index, :], layer.log_variance[output_index, :]])
        jacobian = jax.jacfwd(lambda p, oi=output_index: joint_row_map(p, oi))(params0)
        eigenvalues = np.linalg.eigvals(np.asarray(jacobian, dtype=np.float64))
        radii.append(float(np.max(np.abs(eigenvalues))))
    return radii

def run():
    print("Preparing soft-penalty sweep (8 training runs)...", flush=True)
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
    batch = batches[0]
    selected = np.random.default_rng(protocol.training_seed).permutation(
        protocol.training_samples
    )[:protocol.batch_size]
    np.testing.assert_array_equal(np.asarray(batch.inputs), x[selected].astype(np.float32))

    for beta in BETAS:
        print(f"\nBeta={beta:g}", flush=True)
        infer = make_estep(beta)
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
        weight_variance = float(network.layers[0].weight_variance().mean())
        print(f"  Latent variance: {float(state.variances[0].mean()):.6g}")
        print(f"  Predictive variance: {float(predictive.variance.mean()):.6g}")
        print(f"  Variance residual: {float(residual.mean()):.6g}")
        print(f"  Weight variance: {weight_variance:.6g}")
        print(f"  Latent mean error: {lg._relative_error(np.asarray(state.means[0]), z[selected])}")
        print(f"  Weight mean error vs exact(x,y): {metrics.relative_mean_error}")
        print(f"  Weight variance error vs exact(x,y): {metrics.relative_variance_error}")
        print(f"  Weight variance ratio: {metrics.weight_variance_ratio}", flush=True)
        if weight_variance <= STABILITY_THRESHOLD:
            radii = np.asarray(_joint_row_spectral_radii(
                network, batch.inputs, state.means[0], state.variances[0], config, protocol
            ))
            print(f"  Joint row spectral radii: {radii}")
            print(f"  Rows with radius > 1: {int((radii > 1).sum())}/{len(radii)}", flush=True)
        else:
            print("  Stability check skipped: mean weight variance > 0.05", flush=True)


if __name__ == "__main__":
    run()
