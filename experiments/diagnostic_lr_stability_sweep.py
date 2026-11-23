"""Inspect joint mean/log-variance sensitivity at one trained checkpoint."""

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from bpcn.training.gradients import compute_layer_gradients
from bpcn.utils.safe_math import clamp_weight_log_variance


def row_radii(network, batch, protocol, config, mean_lr, logvar_lr):
    layer = network.layers[0]
    d = layer.input_dim

    def update(params, row):
        candidate = layer._replace(
            mean=layer.mean.at[row].set(params[:d]),
            log_variance=layer.log_variance.at[row].set(params[d:]),
        )
        gradients = compute_layer_gradients(
            candidate,
            batch["inputs"], jnp.zeros_like(batch["inputs"]),
            batch["true_z"],
            jnp.full_like(batch["true_z"], dtl._TRUE_Z_VARIANCE),
            data_scale=1.0 / batch["inputs"].shape[0],
            prior_scale=1.0 / protocol.training_samples,
            weight_kl_scale=config.update.hidden_weight_kl_scale,
            mean_kl_scale=config.update.hidden_mean_kl_scale,
        )
        mean = candidate.mean + mean_lr * gradients.mean_gradient
        logvar = clamp_weight_log_variance(
            candidate.log_variance + logvar_lr * gradients.log_variance_gradient
        )
        return jnp.concatenate((mean[row], logvar[row]))

    radii = []
    for row in range(layer.output_dim):
        params = jnp.concatenate((layer.mean[row], layer.log_variance[row]))
        jacobian = jax.jacfwd(lambda p: update(p, row))(params)
        eigenvalues = np.linalg.eigvals(np.asarray(jacobian, dtype=np.float64))
        radii.append(float(np.abs(eigenvalues).max()))
    return np.asarray(radii)


def run():
    print("Preparing full-batch checkpoint...", flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    config = lg._build_bpcn_config(spec, protocol)
    x, z, y, *_ = dtl._make_dataset_with_latent(spec, protocol)
    target_variance = lg._resolve_target_variance(spec, protocol)
    batch = {
        "inputs": jnp.asarray(x, dtype=jnp.float32),
        "true_z": jnp.asarray(z, dtype=jnp.float32),
        "target_mean": jnp.asarray(y, dtype=jnp.float32),
        "target_variance": jnp.full(y.shape, target_variance, dtype=jnp.float32),
    }
    network, _ = lg._build_network(config, spec, protocol)
    step = dtl._make_bypass_step(config, protocol)
    key = jax.random.PRNGKey(protocol.training_seed)

    for epoch in range(1, 501):
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
            print(f"Checkpoint training: epoch {epoch}/500", flush=True)

    mean_lr = config.update.hidden_mean_learning_rate
    logvar_lr = config.update.hidden_log_variance_learning_rate
    baseline = row_radii(network, batch, protocol, config, mean_lr, logvar_lr)
    print(f"\nBaseline: mean_lr={mean_lr}, logvar_lr={logvar_lr}")
    print(f"Row radii: {baseline}", flush=True)

    for vary, values in (
        ("logvar", (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01)),
        ("mean", (0.05, 0.04, 0.03, 0.02, 0.01, 0.005, 0.0025, 0.001)),
    ):
        print(f"\nVarying {vary} learning rate; other rate fixed", flush=True)
        passing = []
        for value in values:
            radii = row_radii(
                network, batch, protocol, config,
                value if vary == "mean" else mean_lr,
                value if vary == "logvar" else logvar_lr,
            )
            if np.all(radii < 1.0):
                passing.append(value)
            print(
                f"  lr={value:g}  max_radius={radii.max():.6f}  "
                f"rows >= 1: {int((radii >= 1).sum())}/{len(radii)}",
                flush=True,
            )
        print("Largest tested rate with all radii < 1:", max(passing, default=None))


if __name__ == "__main__":
    run()
