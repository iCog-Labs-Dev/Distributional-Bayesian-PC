"""Sweep mean learning rates across Stage B at beta=2; use repository gates."""

from dataclasses import replace

import experiments.linear_gaussian as lg
from experiments.diagnostic_estep_soft_penalty import make_estep
from experiments.diagnostic_soft_penalty_generalization import train_cell

LEARNING_RATES = (0.05, 0.04, 0.03, 0.02, 0.015, 0.01)


def run():
    print("Stage-B LR sweep: beta=2; six rates, twelve cells each", flush=True)
    protocol = lg._Protocol()
    specs = tuple(
        lg.CellSpec("B_hidden", case, noise, mode)
        for case in lg._CASES
        for noise in protocol.output_residual_variances
        for mode in lg._TARGET_VARIANCE_MODES
    )
    datasets = {spec.cell_id: lg._make_dataset(spec, protocol) for spec in specs}
    infer = make_estep(2.0)
    summary = []

    for learning_rate in LEARNING_RATES:
        print(f"\nMean learning rate={learning_rate:g}", flush=True)
        cells = {}
        for index, spec in enumerate(specs, 1):
            print(f"  Cell {index}/{len(specs)}: {spec.cell_id}", flush=True)
            base = lg._build_bpcn_config(spec, protocol)
            config = replace(
                base, update=replace(base.update, hidden_mean_learning_rate=learning_rate)
            )
            if config.inference.initial_mean_perturbation_std != 0:
                raise ValueError("This diagnostic requires initial_mean_perturbation_std=0")
            target_variance = lg._resolve_target_variance(spec, protocol)
            result = train_cell(
                infer, config, spec, datasets[spec.cell_id], target_variance, protocol
            )
            cells[spec.cell_id] = result
            print(f"  Result: {result.status}; epochs={result.epochs}", flush=True)
            if result.metrics is not None:
                metrics = result.metrics
                print(
                    f"    mean_error={metrics.relative_mean_error}; "
                    f"variance_error={metrics.relative_variance_error}; "
                    f"coverage_95={metrics.coverage_95}; "
                    f"variance_ratio={metrics.weight_variance_ratio}", flush=True,
                )

        gates = lg._evaluate_gates(cells, protocol)
        classification = lg._classify(gates, cells, protocol)
        print("  Stage-B gates:", gates["B_hidden"], flush=True)
        print("  Repository classification (Stage-B cells only):", classification, flush=True)
        summary.append((learning_rate, classification))

    print("\nClassification by learning rate (Stage-B cells only):", flush=True)
    for learning_rate, classification in summary:
        print(f"  lr={learning_rate:g}: {classification}", flush=True)


if __name__ == "__main__":
    run()
