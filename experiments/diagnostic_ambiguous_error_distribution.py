from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_ambiguous_regression_v2 as amb
from bpcn.models.moments import moment_forward
from experiments._artifacts import atomic_write_json

_LOG = logging.getLogger(__name__)
_SCHEMA_VERSION = 1

_REPRESENTATIVE_UNITS = (0, 1, 2)
_QUANTILE_POINTS = tuple(i / 10 for i in range(11))  # 0%, 10%, ..., 100%

OUTPUT_PATH = Path("runs/diagnostics/ambiguous_error_distribution/results.json")


def _per_unit_error_distribution(network, config, protocol, batches, e_step_fn):
    all_errors = []
    for batch in batches:
        key = jax.random.PRNGKey(protocol.training_seed)
        latents = e_step_fn(network, batch.inputs, batch.targets, config, key)
        m_z = latents.means[0]
        predictive = moment_forward(network.layers[0], batch.inputs, jnp.zeros_like(batch.inputs))
        e = np.asarray(m_z - predictive.mean, dtype=np.float64)
        all_errors.append(e)
    e_full = np.concatenate(all_errors, axis=0)  # (N, output_dim)
    effective_n = e_full.shape[0]

    e_bar = e_full.mean(axis=0)
    e_bar_sq = e_bar**2
    s_sq = e_full.var(axis=0)
    mean_e2 = (e_full**2).mean(axis=0)
    mean_e3 = (e_full**3).mean(axis=0)
    mean_e4 = (e_full**4).mean(axis=0)
    identity_check = np.abs((e_bar_sq + s_sq) - mean_e2)
    kurt = mean_e4 / (mean_e2**2)
    skew = mean_e3 / (mean_e2**1.5)

    quantiles = {
        f"unit_{unit}": np.quantile(e_full[:, unit], _QUANTILE_POINTS).tolist()
        for unit in _REPRESENTATIVE_UNITS
    }

    return {
        "effective_n": effective_n,
        "e_bar_mean_over_units": float(e_bar.mean()),
        "e_bar_sq_mean_over_units": float(e_bar_sq.mean()),
        "s_sq_mean_over_units": float(s_sq.mean()),
        "mean_e2_mean_over_units": float(mean_e2.mean()),
        "identity_check_max_abs_error": float(identity_check.max()),
        "kurtosis_mean_over_units": float(kurt.mean()),
        "kurtosis_std_over_units": float(kurt.std()),
        "kurtosis_per_unit": kurt.tolist(),
        "skewness_mean_over_units": float(skew.mean()),
        "skewness_per_unit": skew.tolist(),
        "fraction_units_kurt_below_2": float((kurt < 2.0).mean()),
        "fraction_units_kurt_above_2p5": float((kurt > 2.5).mean()),
        "quantiles_e_by_representative_unit": quantiles,
    }


def run() -> Mapping[str, Any]:
    protocol = amb.dtl._PROTOCOL

    _LOG.info("=== calibration (reusing diagnostic_ambiguous_regression_v2's procedure) ===")
    ambiguous_calib = amb._run_condition(protocol, amb._CALIBRATION_SEED_OFFSET, True, amb._BASE_HIDDEN_VARIANCE)
    target_e2 = ambiguous_calib["e_mean_sq"]
    matched_hidden_variance, calibration_attempts = amb._calibrate_noise_matched_hidden_variance(
        protocol, target_e2
    )
    best_attempt = min(calibration_attempts, key=lambda a: abs(a["e_mean_sq"] - target_e2))
    achieved_ratio = best_attempt["e_mean_sq"] / target_e2
    within_tolerance = abs(achieved_ratio - 1.0) <= amb._E2_MATCH_TOLERANCE
    _LOG.info(
        "chosen noise-matched hidden_variance=%.4g achieved_e2_ratio=%.4f within_10pct=%s",
        matched_hidden_variance,
        achieved_ratio,
        within_tolerance,
    )

    results = []
    for seed_offset in amb._SEED_TRIALS:
        trial = {"seed_offset": seed_offset, "conditions": {}}
        for condition_name, ambiguous, hidden_variance in (
            ("ambiguous", True, amb._BASE_HIDDEN_VARIANCE),
            ("clean", False, amb._BASE_HIDDEN_VARIANCE),
            ("noise_matched_clean", False, matched_hidden_variance),
        ):
            dataset = amb._make_dataset(protocol, seed_offset, ambiguous, hidden_variance)
            network, status, epochs, batches, config = amb._train_one(dataset, protocol)
            scalar_measured = amb._measure(network, config, protocol, batches)
            e_step_fn = amb.dsp.make_soft_penalty_estep(amb._BETA)
            distribution = _per_unit_error_distribution(network, config, protocol, batches, e_step_fn)
            trial["conditions"][condition_name] = {
                "status": status,
                "epochs": epochs,
                "hidden_variance_used": hidden_variance,
                **scalar_measured,
                **distribution,
            }
            _LOG.info(
                "seed=%d %-20s status=%-13s weight_var=%.5g mean_e2=%.5g(check vs scalar e2=%.5g) "
                "kurt=%.3f+-%.3f skew=%.3f frac_bimodal(kurt<2)=%.2f frac_gaussian(kurt>2.5)=%.2f "
                "identity_err=%.2e",
                seed_offset,
                condition_name,
                status,
                scalar_measured["weight_variance"],
                distribution["mean_e2_mean_over_units"],
                scalar_measured["e_mean_sq"],
                distribution["kurtosis_mean_over_units"],
                distribution["kurtosis_std_over_units"],
                distribution["skewness_mean_over_units"],
                distribution["fraction_units_kurt_below_2"],
                distribution["fraction_units_kurt_above_2p5"],
                distribution["identity_check_max_abs_error"],
            )
        results.append(trial)

    kurt_ambiguous = [t["conditions"]["ambiguous"]["kurtosis_mean_over_units"] for t in results]
    kurt_matched = [t["conditions"]["noise_matched_clean"]["kurtosis_mean_over_units"] for t in results]
    kurt_clean = [t["conditions"]["clean"]["kurtosis_mean_over_units"] for t in results]
    mean_kurt_ambiguous = float(np.mean(kurt_ambiguous))
    mean_kurt_matched = float(np.mean(kurt_matched))
    mean_kurt_clean = float(np.mean(kurt_clean))

    separates = (mean_kurt_ambiguous < 2.0) and (mean_kurt_matched > 2.5)
    verdict_label = "H5_SUPPORTED" if separates else "H5_REFUTED"
    _LOG.info(
        "VERDICT: %s | mean kurtosis: ambiguous=%.3f clean=%.3f noise_matched_clean=%.3f",
        verdict_label,
        mean_kurt_ambiguous,
        mean_kurt_clean,
        mean_kurt_matched,
    )

    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": "diagnostic_ambiguous_error_distribution",
        "calibration": {
            "ambiguous_e2_target": target_e2,
            "chosen_hidden_variance": matched_hidden_variance,
            "achieved_e2_ratio_to_target": achieved_ratio,
            "within_10pct_tolerance": within_tolerance,
            "candidates": calibration_attempts,
        },
        "trials": results,
        "summary": {
            "mean_kurtosis_ambiguous": mean_kurt_ambiguous,
            "mean_kurtosis_clean": mean_kurt_clean,
            "mean_kurtosis_noise_matched_clean": mean_kurt_matched,
            "verdict": verdict_label,
        },
    }


def save_result(result: Mapping[str, Any], path: Path = OUTPUT_PATH) -> Path:
    atomic_write_json(path, result)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run()
    destination = save_result(result)
    _LOG.info("saved diagnostic result to %s", destination)


if __name__ == "__main__":
    main()
