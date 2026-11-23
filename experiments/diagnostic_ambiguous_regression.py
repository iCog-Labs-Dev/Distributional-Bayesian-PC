"""Paired clean/sign-mixture regression; compare learned variance and Gaussian NLL."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_soft_penalty import make_estep
from experiments.diagnostic_estep_variance_ablation import train
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.models.moments import moment_forward

_BETA = 2.0
SPEC = lg.CellSpec("A_output", "independent", 0.01, "matched")


def make_data(protocol, seed, ambiguous):
    rng = np.random.default_rng(protocol.data_seed + seed)
    signs = np.random.default_rng(protocol.data_seed + seed + 1_000_000)
    d = protocol.dimension
    weight = rng.standard_normal((d, d)) * protocol.prior_std

    def emit(x):
        z = x + np.sqrt(protocol.stage_a_hidden_residual_variance) * rng.standard_normal(x.shape)
        noise = np.sqrt(SPEC.output_residual_variance) * rng.standard_normal(x.shape)
        sign = signs.choice([-1.0, 1.0], size=(len(x), 1)) if ambiguous else 1.0
        return sign * (z @ weight.T) + noise

    x = rng.standard_normal((protocol.training_samples, d))
    y = emit(x)
    test_x = rng.standard_normal((protocol.test_samples, d))
    test_y = emit(test_x)
    return lg._SyntheticData(
        train_inputs=x, train_targets=y, test_inputs=test_x,
        test_targets=test_y, true_weight=weight,
    )


def _measure(network, config, protocol, batches, dataset):
    batch0 = batches[0]
    e_step_fn = make_estep(_BETA)
    key = jax.random.PRNGKey(protocol.training_seed)
    latents = e_step_fn(network, batch0.inputs, batch0.targets, config, key)
    from bpcn.models.activations import activation_moments
    feature_mean, feature_variance = activation_moments(network.hidden_activations[-1], latents.top_mean, latents.top_variance)
    predictive = moment_forward(network.output_layer, feature_mean, feature_variance)
    terms = gaussian_kl(batch0.target_mean, batch0.target_variance, predictive.mean, predictive.variance)
    v_p = float(np.asarray(predictive.variance).mean())
    v_z_proxy = float(np.asarray(batch0.target_variance).mean())
    r = float(np.asarray(terms.variance_residual).mean())
    e_mean_sq = float((np.asarray(terms.mean_error) ** 2).mean())
    v_z_minus_v_p = v_z_proxy - v_p
    positive_r_fraction = float((np.asarray(terms.variance_residual) > 0).mean())
    weight_variance = float(np.asarray(network.output_layer.weight_variance(), dtype=np.float64).mean())
    weight_mean_norm = float(np.linalg.norm(np.asarray(network.output_layer.mean)))
    true_weight_norm = float(np.linalg.norm(dataset.true_weight))
    from bpcn.models.activations import activation_moments as _activation_moments
    test_inputs_jax = jnp.asarray(dataset.test_inputs, dtype=jnp.float32)
    hidden_predictive = moment_forward(network.layers[0], test_inputs_jax, jnp.zeros_like(test_inputs_jax))
    test_feature_mean, test_feature_variance = _activation_moments(network.hidden_activations[0], hidden_predictive.mean, hidden_predictive.variance)
    test_predictive = moment_forward(network.output_layer, test_feature_mean, test_feature_variance)
    test_mean = np.asarray(test_predictive.mean, dtype=np.float64)
    test_variance = np.asarray(test_predictive.variance, dtype=np.float64)
    residual = dataset.test_targets - test_mean
    nll = float((0.5 * (np.log(2 * np.pi * test_variance) + residual ** 2 / test_variance)).sum(axis=1).mean())
    return {'weight_variance': weight_variance, 'weight_mean_norm': weight_mean_norm, 'true_weight_norm': true_weight_norm, 'v_p': v_p, 'v_z_proxy': v_z_proxy, 'r': r, 'e_mean_sq': e_mean_sq, 'v_z_minus_v_p': v_z_minus_v_p, 'positive_r_fraction': positive_r_fraction, 'kappa': v_z_proxy / v_p, 'test_predictive_nll': nll}

def run():
    print("Paired regression test: 3 data seeds x 2 conditions = 6 runs", flush=True)
    protocol = lg._Protocol()
    config = lg._build_bpcn_config(SPEC, protocol)
    config = replace(config, update=replace(config.update, output_mean_learning_rate=0.05))
    if config.inference.initial_mean_perturbation_std != 0:
        raise ValueError("This diagnostic requires initial_mean_perturbation_std=0")
    target_variance = lg._resolve_target_variance(SPEC, protocol)
    infer = make_estep(_BETA)
    ratios = []

    for seed in (0, 1, 2):
        pair = {}
        for label, ambiguous in (("ambiguous", True), ("clean", False)):
            print(f"\nData seed offset={seed}; condition={label}", flush=True)
            data = make_data(protocol, seed, ambiguous)
            if ambiguous:
                paired_inputs = data.train_inputs
                paired_weight = data.true_weight
            else:
                np.testing.assert_array_equal(data.train_inputs, paired_inputs)
                np.testing.assert_array_equal(data.true_weight, paired_weight)
            batches = lg._batches(data, target_variance, protocol)
            network, status, epochs = train(infer, config, SPEC, batches, protocol)
            print(f"  {status}; epochs={epochs}", flush=True)
            if status == "NON_FINITE":
                continue
            measured = _measure(network, config, protocol, batches, data)
            pair[label] = (status, measured["weight_variance"])
            for name, value in measured.items():
                print(f"  {name}: {value}", flush=True)
        if "ambiguous" in pair and "clean" in pair:
            ratio = pair["ambiguous"][1] / pair["clean"][1]
            print(f"  Weight-variance ratio (ambiguous/clean): {ratio:.6g}", flush=True)
            if all(status == "COMPLETE" for status, _ in pair.values()):
                ratios.append(ratio)

    print(f"\nConverged pairs: {len(ratios)}/3", flush=True)
    if ratios:
        print(f"Weight-variance ratio: mean={np.mean(ratios):.6g}, SD={np.std(ratios):.6g}", flush=True)


if __name__ == "__main__":
    run()
