"""Hidden-layer sign-mixture test with a clean control matched on inferred error."""

import math

import jax
import jax.numpy as jnp
import numpy as np

import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_soft_penalty import make_estep
from experiments.diagnostic_estep_variance_ablation import train
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.models.moments import moment_forward

_BETA = 2.0
_SPEC = lg.CellSpec("B_hidden", "independent", 0.01, "matched")
_Dataset = lg._SyntheticData
CANDIDATES = (0.01, 1.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0)


def _make_dataset(protocol, seed_offset, ambiguous, hidden_variance):
    data_seed = protocol.data_seed + seed_offset
    rng = np.random.default_rng(data_seed)
    dimension = protocol.dimension
    true_weight = rng.standard_normal((dimension, dimension)) * protocol.prior_std
    output_variance = _SPEC.output_residual_variance
    sign_rng = np.random.default_rng(data_seed + 1000000)

    def emit(inputs: np.ndarray) -> np.ndarray:
        sample_count = len(inputs)
        hidden_noise = math.sqrt(hidden_variance) * rng.standard_normal((sample_count, dimension))
        output_noise = math.sqrt(output_variance) * rng.standard_normal((sample_count, dimension))
        if ambiguous:
            sign = sign_rng.choice([-1.0, 1.0], size=(sample_count, 1))
        else:
            sign = 1.0
        z = sign * (inputs @ true_weight.T) + hidden_noise
        return z + output_noise
    train_inputs = rng.standard_normal((protocol.training_samples, dimension))
    train_targets = emit(train_inputs)
    test_inputs = rng.standard_normal((protocol.test_samples, dimension))
    test_targets = emit(test_inputs)
    return _Dataset(train_inputs=train_inputs.astype(np.float64), train_targets=train_targets.astype(np.float64), test_inputs=test_inputs.astype(np.float64), test_targets=test_targets.astype(np.float64), true_weight=true_weight.astype(np.float64))

def _measure(network, config, protocol, batches):
    batch0 = batches[0]
    e_step_fn = make_estep(_BETA)
    key = jax.random.PRNGKey(protocol.training_seed)
    latents = e_step_fn(network, batch0.inputs, batch0.targets, config, key)
    m_z, v_z = (latents.means[0], latents.variances[0])
    predictive = moment_forward(network.layers[0], batch0.inputs, jnp.zeros_like(batch0.inputs))
    terms = gaussian_kl(m_z, v_z, predictive.mean, predictive.variance)
    v_p = float(np.asarray(predictive.variance).mean())
    v_z_mean = float(np.asarray(v_z).mean())
    r = float(np.asarray(terms.variance_residual).mean())
    e_mean_sq = float((np.asarray(terms.mean_error) ** 2).mean())
    v_z_minus_v_p = v_z_mean - v_p
    weight_variance = float(np.asarray(network.layers[0].weight_variance(), dtype=np.float64).mean())
    alpha2 = protocol.prior_std ** 2
    return {'weight_variance': weight_variance, 'sigma2_over_alpha2': weight_variance / alpha2, 'v_p': v_p, 'v_z': v_z_mean, 'r': r, 'r_sign': 'positive' if r > 0 else 'negative' if r < 0 else 'zero', 'e_mean_sq': e_mean_sq, 'v_z_minus_v_p': v_z_minus_v_p, 'kappa': v_z_mean / v_p}

def run():
    print("Hidden-layer sign-mixture test with noise-control calibration...", flush=True)
    protocol = lg._Protocol()
    config = lg._build_bpcn_config(_SPEC, protocol)
    if config.inference.initial_mean_perturbation_std != 0:
        raise ValueError("This diagnostic requires initial_mean_perturbation_std=0")
    target_variance = lg._resolve_target_variance(_SPEC, protocol)
    infer = make_estep(_BETA)
    cache = {}

    def condition(seed, ambiguous, variance):
        cache_key = (seed, ambiguous, variance)
        if cache_key in cache:
            print(f"Reusing seed={seed}, sign flips={ambiguous}, noise variance={variance}", flush=True)
            return cache[cache_key]
        print(f"\nTraining seed={seed}, sign flips={ambiguous}, noise variance={variance}", flush=True)
        data = _make_dataset(protocol, seed, ambiguous, variance)
        batches = lg._batches(data, target_variance, protocol)
        network, status, epochs = train(infer, config, _SPEC, batches, protocol)
        result = dict(status=status, epochs=epochs, hidden_variance_used=variance)
        print(f"  {status}; epochs={epochs}", flush=True)
        if status != "NON_FINITE":
            result.update(_measure(network, config, protocol, batches))
            for name, value in result.items():
                print(f"  {name}: {value}", flush=True)
        cache[cache_key] = result
        return result

    target = condition(0, True, 0.01)
    if target["status"] != "COMPLETE":
        print("Calibration stopped: ambiguous reference did not converge.", flush=True)
        return
    target_e2 = target["e_mean_sq"]
    if not np.isfinite(target_e2) or target_e2 <= 0:
        print("Calibration stopped: target inference error must be finite and positive.", flush=True)
        return
    candidates = [condition(0, False, variance) for variance in CANDIDATES]
    candidates = [r for r in candidates if r["status"] == "COMPLETE" and np.isfinite(r["e_mean_sq"])]
    if not candidates:
        print("Calibration stopped: no finite, converged clean candidate.", flush=True)
        return
    best = min(candidates, key=lambda r: abs(r["e_mean_sq"] - target_e2))
    matched_variance = best["hidden_variance_used"]
    match_ratio = best["e_mean_sq"] / target_e2
    print(f"\nSelected clean noise variance: {matched_variance}")
    print(f"Calibration e² ratio (clean/ambiguous): {match_ratio:.6g}")
    print(f"Within 10%: {abs(match_ratio - 1) <= 0.10}", flush=True)

    ratios = []
    ambiguous_errors, matched_errors = [], []
    for seed in (0, 1, 2):
        print(f"\nComparison for data seed offset={seed}", flush=True)
        ambiguous = condition(seed, True, 0.01)
        clean = condition(seed, False, 0.01)
        matched = condition(seed, False, matched_variance)
        if any(r["status"] == "NON_FINITE" for r in (ambiguous, clean, matched)):
            print("Pair comparison unavailable: non-finite training result.", flush=True)
            continue
        print(f"  Weight variance: ambiguous/clean={ambiguous['weight_variance']/clean['weight_variance']:.6g}")
        ratio = ambiguous["weight_variance"] / matched["weight_variance"]
        print(f"  Weight variance: ambiguous/noise-control={ratio:.6g}")
        e2_ratio = matched["e_mean_sq"] / ambiguous["e_mean_sq"] if ambiguous["e_mean_sq"] > 0 else np.nan
        print(f"  e²: noise-control/ambiguous={e2_ratio:.6g}; within 10%={abs(e2_ratio-1) <= 0.10}", flush=True)
        if ambiguous["status"] == matched["status"] == "COMPLETE":
            ratios.append(ratio)
            ambiguous_errors.append(ambiguous["e_mean_sq"])
            matched_errors.append(matched["e_mean_sq"])

    print(f"\nConverged ambiguous/noise-control pairs: {len(ratios)}/3", flush=True)
    if ratios:
        print(f"Weight-variance ratio: mean={np.mean(ratios):.6g}, SD={np.std(ratios):.6g}")
        if np.mean(matched_errors) > 0:
            print(f"Mean e² ratio (ambiguous/noise-control): {np.mean(ambiguous_errors)/np.mean(matched_errors):.6g}", flush=True)


if __name__ == "__main__":
    run()
