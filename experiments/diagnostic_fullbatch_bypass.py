"""Train with true latents and one full batch; compare with exact p(W | x,z)."""

import jax.numpy as jnp
import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg


def run():
    print("Preparing full-batch diagnostic...", flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    config = lg._build_bpcn_config(spec, protocol)
    target_variance = lg._resolve_target_variance(spec, protocol)

    x, z, y, test_x, _, test_y, true_weight = dtl._make_dataset_with_latent(
        spec, protocol
    )
    dtl._verify_dataset_match(x, y, test_x, test_y, true_weight, protocol)
    _, exact_xz = dtl._exact_references(x, z, y, true_weight, protocol)
    batch = {
        "inputs": jnp.asarray(x, dtype=jnp.float32),
        "true_z": jnp.asarray(z, dtype=jnp.float32),
        "target_mean": jnp.asarray(y, dtype=jnp.float32),
        "target_variance": jnp.full(y.shape, target_variance, dtype=jnp.float32),
    }

    print(f"Training with all {len(x)} samples per batch...", flush=True)
    network, _ = lg._build_network(config, spec, protocol)
    network, status, epochs = dtl.train_bypass(
        network, config, (batch,), protocol
    )
    print(f"Status: {status}; epochs: {epochs}", flush=True)
    if status == "NON_FINITE":
        return

    data = lg._SyntheticData(
        train_inputs=x, train_targets=y,
        test_inputs=test_x, test_targets=test_y,
        true_weight=true_weight,
    )
    metrics = lg._cell_metrics(network, spec, exact_xz, data, protocol)
    print(f"Mean error:     {metrics.relative_mean_error}")
    print(f"Variance error: {metrics.relative_variance_error}")
    print(f"Variance ratio: {metrics.weight_variance_ratio}")


if __name__ == "__main__":
    run()
