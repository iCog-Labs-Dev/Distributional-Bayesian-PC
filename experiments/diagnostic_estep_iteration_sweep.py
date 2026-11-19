from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

import experiments.diagnostic_estep_soft_penalty as dsp
import experiments.diagnostic_estep_variance_ablation as dva
import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg
from bpcn.losses.distributional_kl import gaussian_kl
from bpcn.models.moments import moment_forward
from experiments._artifacts import atomic_write_json

_LOG = logging.getLogger(__name__)
_SCHEMA_VERSION = 1

_BETA = 2.0
_T_Z_VALUES = (20, 40, 100, 200)
_CELLS = (
    lg.CellSpec("B_hidden", "independent", 0.01, "matched"),
    lg.CellSpec("B_hidden", "duplicate_agree", 0.01, "matched"),
)

OUTPUT_PATH = Path("runs/diagnostics/estep_iteration_sweep/results.json")


def _tz_config(spec, protocol, t_z: int):
    base = lg._build_bpcn_config(spec, protocol)
    return replace(base, inference=replace(base.inference, steps=t_z))


def run() -> Mapping[str, Any]:
    protocol = dtl._PROTOCOL
    e_step_fn = dsp.make_soft_penalty_estep(_BETA)

    results = []
    for spec in _CELLS:
        target_variance = lg._resolve_target_variance(spec, protocol)
        (
            train_inputs,
            train_hidden,
            train_targets,
            test_inputs,
            test_hidden,
            test_targets,
            true_weight,
        ) = dtl._make_dataset_with_latent(spec, protocol)
        synthetic_data = lg._SyntheticData(
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            true_weight=true_weight,
        )
        noise_covariance_xy = lg._noise_covariance(spec, true_weight, protocol)
        exact_xy = lg._exact_posterior(
            train_inputs, train_targets, noise_covariance_xy, protocol.prior_std
        )

        for t_z in _T_Z_VALUES:
            config = _tz_config(spec, protocol, t_z)
            start = time.perf_counter()
            network, status, epochs, _history_tail, batches = dva._train_with_custom_estep(
                e_step_fn, config, spec, synthetic_data, target_variance, protocol
            )
            elapsed = time.perf_counter() - start
            batch0 = batches[0]
            key = jax.random.PRNGKey(protocol.training_seed)
            latents = e_step_fn(network, batch0.inputs, batch0.targets, config, key)
            m_z, v_z = latents.means[0], latents.variances[0]
            predictive = moment_forward(network.layers[0], batch0.inputs, jnp.zeros_like(batch0.inputs))
            terms = gaussian_kl(m_z, v_z, predictive.mean, predictive.variance)
            e_mean_sq = float((np.asarray(terms.mean_error) ** 2).mean())
            v_p = float(np.asarray(predictive.variance).mean())
            v_z_mean = float(np.asarray(v_z).mean())
            kappa = v_z_mean / v_p
            weight_variance = float(
                np.asarray(network.layers[0].weight_variance(), dtype=np.float64).mean()
            )
            metrics = lg._cell_metrics(network, spec, exact_xy, synthetic_data, protocol)
            row = {
                "cell_id": spec.cell_id,
                "t_z": t_z,
                "status": status,
                "epochs": epochs,
                "wall_clock_seconds": elapsed,
                "wall_clock_per_epoch": elapsed / epochs if epochs else None,
                "e_mean_sq": e_mean_sq,
                "kappa": kappa,
                "v_z_mean": v_z_mean,
                "v_p": v_p,
                "weight_variance": weight_variance,
                "relative_mean_error": metrics.relative_mean_error,
                "relative_variance_error": metrics.relative_variance_error,
                "coverage_95": metrics.coverage_95,
            }
            results.append(row)
            _LOG.info(
                "%-45s T_z=%4d status=%-13s epochs=%4d e_mean_sq=%.5g kappa=%.4f "
                "weight_var=%.5g var_err=%s sec/epoch=%.4f",
                spec.cell_id,
                t_z,
                status,
                epochs,
                e_mean_sq,
                kappa,
                weight_variance,
                f"{metrics.relative_variance_error:.4g}",
                row["wall_clock_per_epoch"],
            )

    return {
        "schema_version": _SCHEMA_VERSION,
        "experiment": "diagnostic_estep_iteration_sweep",
        "beta": _BETA,
        "t_z_values": _T_Z_VALUES,
        "results": results,
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
