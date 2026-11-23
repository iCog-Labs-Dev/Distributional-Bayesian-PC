"""Probe averaged weight-variance gradients with frozen or rerun latent inference."""

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_soft_penalty import make_estep
from experiments.diagnostic_estep_variance_ablation import train
from bpcn.training.gradients import compute_layer_gradients
from bpcn.utils.safe_math import clamp_weight_log_variance

GRID = np.logspace(-6, -1, 16)


def crossings(x, values):
    """Approximate scalar zero crossings by interpolation in log variance."""
    x, values = np.asarray(x), np.asarray(values)
    roots = [float(a) for a, b in zip(x, values) if b == 0]
    for i in range(len(x) - 1):
        a, b = values[i:i + 2]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            fraction = -a / (b - a)
            roots.append(float(np.exp(np.log(x[i]) + fraction * np.log(x[i + 1] / x[i]))))
    return sorted(set(roots))


def gradient_point(layer, batch, state, config, protocol):
    terms = compute_layer_gradients(
        layer, batch.inputs, jnp.zeros_like(batch.inputs),
        state.means[0], state.variances[0],
        data_scale=1.0 / batch.inputs.shape[0],
        prior_scale=1.0 / protocol.training_samples,
        weight_kl_scale=config.update.hidden_weight_kl_scale,
        mean_kl_scale=config.update.hidden_mean_kl_scale,
    )
    data = np.asarray(terms.log_variance_data_gradient, dtype=np.float64)
    prior = np.asarray(terms.log_variance_prior_gradient, dtype=np.float64)
    vp = float(terms.predictive.variance.mean())
    vz = float(state.variances[0].mean())
    e2 = float((terms.mean_error**2).mean())
    residual = float(terms.variance_residual.mean())
    print(f"    Vp={vp:.5g} Vz={vz:.5g} rV={residual:.5g} E[e²]={e2:.5g} Vz-Vp={vz-vp:.5g}")
    print(
        f"    mean g_data={data.mean():.5g} mean g_prior={prior.mean():.5g} "
        f"mean total={(data+prior).mean():.5g} mean |total|={np.abs(data+prior).mean():.5g}",
        flush=True,
    )
    return float(data.mean()), float(prior.mean())


def run():
    print("Training reference cell with beta=2...", flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    config = lg._build_bpcn_config(spec, protocol)
    if config.inference.initial_mean_perturbation_std != 0:
        raise ValueError("This diagnostic requires initial_mean_perturbation_std=0")
    target_variance = lg._resolve_target_variance(spec, protocol)
    data = lg._make_dataset(spec, protocol)
    batches = lg._batches(data, target_variance, protocol)
    infer = make_estep(2.0)
    network, status, epochs = train(infer, config, spec, batches, protocol)
    print(f"Reference training: {status}; epochs={epochs}", flush=True)
    if status == "NON_FINITE":
        return
    if status != "COMPLETE":
        print("Reference did not converge; interpret the probes accordingly.", flush=True)

    batch = batches[0]
    key = jax.random.PRNGKey(protocol.training_seed)
    frozen_state = infer(network, batch.inputs, batch.targets, config, key)
    base_layer = network.layers[0]
    observed = float(base_layer.weight_variance().mean())
    covariance = lg._noise_covariance(spec, data.true_weight, protocol)
    exact = lg._exact_posterior(
        data.train_inputs, data.train_targets, covariance, protocol.prior_std
    )
    exact_variance = float(exact.diagonal_covariance.mean())
    oracle_layer = base_layer._replace(mean=jnp.asarray(exact.mean, dtype=base_layer.mean.dtype))
    print(f"Trained mean weight variance: {observed:.6g}")
    print(f"Exact(x,y) mean diagonal variance: {exact_variance:.6g}")
    print(
        f"N={protocol.training_samples}; batch={len(batch.inputs)}; "
        f"prior variance={protocol.prior_std**2}; "
        f"E[sum x²]={float((batch.inputs**2).sum(axis=1).mean()):.6g}", flush=True,
    )
    print("\nGradient at trained weights:", flush=True)
    gradient_point(base_layer, batch, frozen_state, config, protocol)

    for label, layer, rerun, seed in (
        ("1a: frozen latents, trained means", base_layer, False, protocol.training_seed),
        ("1b: rerun E-step, trained means", base_layer, True, protocol.training_seed),
        ("1e: rerun E-step, exact(x,y) means", oracle_layer, True, protocol.training_seed + 1),
    ):
        print(f"\n{label}", flush=True)
        variances, data_gradients, prior_gradients = [], [], []
        for index, sigma2 in enumerate(GRID):
            candidate = layer._replace(log_variance=clamp_weight_log_variance(
                jnp.full_like(layer.log_variance, np.log(sigma2))
            ))
            candidate_network = network._replace(layers=(candidate,) + network.layers[1:])
            state = frozen_state
            if rerun:
                state = infer(
                    candidate_network, batch.inputs, batch.targets, config,
                    jax.random.fold_in(jax.random.PRNGKey(seed), index),
                )
            actual_variance = float(candidate.weight_variance().mean())
            print(f"  sigma²={actual_variance:.6g}", flush=True)
            gd, gp = gradient_point(candidate, batch, state, config, protocol)
            variances.append(actual_variance)
            data_gradients.append(gd)
            prior_gradients.append(gp)
        total = np.asarray(data_gradients) + np.asarray(prior_gradients)
        print("  Zeros of averaged data+prior gradient:", crossings(variances, total))
        print("  Zeros of averaged data gradient only:", crossings(variances, data_gradients), flush=True)


if __name__ == "__main__":
    run()
