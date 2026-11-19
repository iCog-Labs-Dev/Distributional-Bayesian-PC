"""Test beta=2 across Stage B; use the repository's existing gate routines."""

import json
from pathlib import Path

import jax
import numpy as np

import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_soft_penalty import make_estep
from experiments.diagnostic_estep_variance_ablation import make_step

BETA = 2.0
BASELINE = Path("runs/linear_gaussian/results.json")


def train_cell(infer, config, spec, data, target_variance, protocol):
    network, frozen_index = lg._build_network(config, spec, protocol)
    frozen_mean = np.asarray(network.layers[frozen_index].mean).copy()
    frozen_logvar = np.asarray(network.layers[frozen_index].log_variance).copy()
    learned = lg._learned_index(spec.stage)
    batches = lg._batches(data, target_variance, protocol)
    step = make_step(infer, config, protocol)
    key = jax.random.PRNGKey(protocol.training_seed)
    stable = 0
    status = "NOT_CONVERGED"
    history = []
    for epoch in range(1, protocol.max_epochs + 1):
        old_mean = np.asarray(network.layers[learned].mean, dtype=np.float64).copy()
        old_logvar = np.asarray(network.layers[learned].log_variance, dtype=np.float64).copy()
        for batch in batches:
            key, batch_key = jax.random.split(key)
            network = step(
                network, batch.inputs, batch.targets.mean, batch.targets.variance,
                batch.targets.class_indices, batch_key,
            )
        mean = np.asarray(network.layers[learned].mean, dtype=np.float64)
        logvar = np.asarray(network.layers[learned].log_variance, dtype=np.float64)
        if not (np.isfinite(mean).all() and np.isfinite(logvar).all()):
            status = "NON_FINITE"
            break
        mean_delta = float(np.abs(mean - old_mean).max())
        logvar_delta = float(np.abs(logvar - old_logvar).max())
        history.append(dict(epoch=epoch, mean_delta=mean_delta, log_variance_delta=logvar_delta))
        history = history[-5:]
        stable = stable + 1 if max(mean_delta, logvar_delta) < protocol.convergence_tolerance else 0
        if epoch == 1 or epoch % 100 == 0:
            print(f"  epoch={epoch} mean_delta={mean_delta:.4g} logvar_delta={logvar_delta:.4g}", flush=True)
        if stable >= protocol.convergence_patience:
            status = "COMPLETE"
            break
    lg._assert_frozen_layer_unchanged(network, frozen_index, frozen_mean, frozen_logvar)
    final = history[-1] if history and status != "NON_FINITE" else {}
    covariance = lg._noise_covariance(spec, data.true_weight, protocol)
    exact = lg._exact_posterior(
        data.train_inputs, data.train_targets, covariance, protocol.prior_std
    )
    metrics = None if status == "NON_FINITE" else lg._cell_metrics(
        network, spec, exact, data, protocol
    )
    return lg.CellResult(
        spec=spec, status=status, converged=status == "COMPLETE", epochs=epoch,
        final_mean_delta=final.get("mean_delta"),
        final_log_variance_delta=final.get("log_variance_delta"),
        target_variance=target_variance,
        hidden_residual_variance=lg._hidden_residual_variance(spec.stage, protocol),
        noise_diagonal_mean=float(np.diag(covariance).mean()),
        noise_off_diagonal_max=float(np.abs(covariance - np.diag(np.diag(covariance))).max()),
        metrics=metrics, stationarity=None, history_tail=tuple(history),
    )


def run():
    print("Preparing Stage-B generalization test; beta=2...", flush=True)
    protocol = lg._Protocol()
    baseline = {}
    if BASELINE.exists():
        payload = json.loads(BASELINE.read_text(encoding="utf-8"))
        for cell in payload.get("cells", ()):
            spec_data = cell.get("spec", {})
            if spec_data.get("stage") == "B_hidden":
                spec = lg.CellSpec(
                    spec_data["stage"], spec_data["case"],
                    spec_data["output_residual_variance"], spec_data["target_variance_mode"],
                )
                baseline[spec.cell_id] = cell
    else:
        print("Baseline file absent; previous-run comparisons unavailable.", flush=True)

    infer = make_estep(BETA)
    cells = {}
    for case in lg._CASES:
        for output_variance in protocol.output_residual_variances:
            for target_mode in lg._TARGET_VARIANCE_MODES:
                spec = lg.CellSpec("B_hidden", case, output_variance, target_mode)
                print(f"\n{spec.cell_id}", flush=True)
                config = lg._build_bpcn_config(spec, protocol)
                if config.inference.initial_mean_perturbation_std != 0:
                    raise ValueError("This diagnostic requires initial_mean_perturbation_std=0")
                data = lg._make_dataset(spec, protocol)
                target_variance = lg._resolve_target_variance(spec, protocol)
                result = train_cell(infer, config, spec, data, target_variance, protocol)
                cells[spec.cell_id] = result
                old = baseline.get(spec.cell_id, {})
                old_metrics = old.get("metrics") or {}
                print(f"  Soft penalty: {result.status}; epochs={result.epochs}")
                print(f"  Baseline status: {old.get('status', 'unavailable')}")
                for field in (
                    "relative_mean_error", "relative_variance_error", "coverage_95",
                    "relative_prior_retention_error", "total_precision_dbpcn",
                ):
                    value = getattr(result.metrics, field) if result.metrics is not None else None
                    print(f"  {field}: soft={value}; baseline={old_metrics.get(field)}", flush=True)

    print("\nEvaluating repository gates (Stage B only)...", flush=True)
    gates = lg._evaluate_gates(cells, protocol)
    print(gates["B_hidden"], flush=True)
    classification = lg._classify(gates, cells, protocol)
    print("Repository classification with only Stage-B cells supplied:")
    print(classification, flush=True)


if __name__ == "__main__":
    run()
