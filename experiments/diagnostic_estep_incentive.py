"""Inspect E-step energy and variance gradients before and after training."""

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from bpcn.inference.e_step import infer_latents
from bpcn.inference.shared_energy import latent_free_energy
from bpcn.inference.state import LatentState


def inspect(label, network, batch, true_z, protocol, config, key):
    print(f"\n{label}", flush=True)
    actual = infer_latents(network, batch.inputs, batch.targets, config, key).latents
    noise_variance = lg._hidden_residual_variance(dtl._SPEC.stage, protocol)
    reference = LatentState(
        means=(true_z,),
        variances=(jnp.full_like(true_z, noise_variance),),
    )

    def objective(mean, logvar):
        state = LatentState(means=(mean,), variances=(jnp.exp(logvar),))
        return latent_free_energy(
            network, batch.inputs, state, targets=batch.targets,
            key=key, mc_samples=1,
            output_loss_weight=config.update.output_loss_weight,
        )

    energy_and_grad = jax.value_and_grad(objective, argnums=(0, 1))
    energies = []
    for name, state in (
        ("E-step result", actual),
        ("True-z / hidden-noise reference", reference),
    ):
        energy, (dmean, dlogvar) = energy_and_grad(
            state.means[0], jnp.log(state.variances[0])
        )
        energies.append(float(energy))
        print(f"  {name}")
        print(f"    F: {float(energy):.6g}")
        print(f"    Mean latent variance: {float(state.variances[0].mean()):.6g}")
        print(f"    Mean |dF/dmean|: {float(jnp.abs(dmean).mean()):.6g}")
        print(f"    Mean |dF/dlogvar|: {float(jnp.abs(dlogvar).mean()):.6g}")
        print(f"    Mean dF/dlogvar: {float(dlogvar.mean()):.6g}")
        print(
            f"    Fraction of negative variance gradients: "
            f"{float((dlogvar < 0).mean()):.3f}", flush=True,
        )
    print(f"  F(E-step) - F(reference): {energies[0] - energies[1]:.6g}", flush=True)


def run():
    print("Preparing E-step diagnostic...", flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    config = lg._build_bpcn_config(spec, protocol)
    target_variance = lg._resolve_target_variance(spec, protocol)
    x, z, y, test_x, _, test_y, true_weight = dtl._make_dataset_with_latent(
        spec, protocol
    )
    dtl._verify_dataset_match(x, y, test_x, test_y, true_weight, protocol)
    data = lg._SyntheticData(
        train_inputs=x, train_targets=y, test_inputs=test_x,
        test_targets=test_y, true_weight=true_weight,
    )
    batch = lg._batches(data, target_variance, protocol)[0]
    selected = np.random.default_rng(protocol.training_seed).permutation(
        protocol.training_samples
    )[:protocol.batch_size]
    np.testing.assert_array_equal(np.asarray(batch.inputs), x[selected].astype(np.float32))
    true_z = jnp.asarray(z[selected], dtype=jnp.float32)
    key = jax.random.PRNGKey(protocol.training_seed)

    network, _ = lg._build_network(config, spec, protocol)
    inspect("Initialization", network, batch, true_z, protocol, config, key)
    print("\nStarting NORMAL training...", flush=True)
    outcome = lg._train_cell(
        lg.make_batch_step(config, protocol.training_samples),
        network, spec, data, target_variance, protocol,
    )
    print(f"Training: {outcome.status}; epochs={outcome.epochs}", flush=True)
    if outcome.status != "NON_FINITE":
        inspect("After training", outcome.network, batch, true_z, protocol, config, key)


if __name__ == "__main__":
    run()
