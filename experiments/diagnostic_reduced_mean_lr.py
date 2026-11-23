"""Retest NORMAL and true-z BYPASS with hidden mean learning rate 0.02."""

from dataclasses import replace

import experiments.diagnostic_true_latent as dtl
import experiments.linear_gaussian as lg


def show_result(name, network, status, epochs, reference, spec, data, protocol):
    print(f"\n{name}: {status}; epochs={epochs}", flush=True)
    if status == "NON_FINITE":
        return
    metrics = lg._cell_metrics(network, spec, reference, data, protocol)
    print(f"  Mean error:     {metrics.relative_mean_error}")
    print(f"  Variance error: {metrics.relative_variance_error}")
    print(f"  Variance ratio: {metrics.weight_variance_ratio}", flush=True)


def run():
    print("Preparing reduced-mean-LR retest...", flush=True)
    protocol = lg._Protocol()
    spec = dtl._SPEC
    base = lg._build_bpcn_config(spec, protocol)
    config = replace(
        base, update=replace(base.update, hidden_mean_learning_rate=0.02)
    )
    print(
        f"Mean LR: {base.update.hidden_mean_learning_rate} -> 0.02; "
        f"log-variance LR: {config.update.hidden_log_variance_learning_rate}",
        flush=True,
    )
    target_variance = lg._resolve_target_variance(spec, protocol)
    x, z, y, test_x, _, test_y, true_weight = dtl._make_dataset_with_latent(
        spec, protocol
    )
    dtl._verify_dataset_match(x, y, test_x, test_y, true_weight, protocol)
    exact_xy, exact_xz = dtl._exact_references(x, z, y, true_weight, protocol)
    data = lg._SyntheticData(
        train_inputs=x, train_targets=y, test_inputs=test_x,
        test_targets=test_y, true_weight=true_weight,
    )

    print("Starting NORMAL training...", flush=True)
    network, _ = lg._build_network(config, spec, protocol)
    normal = lg._train_cell(
        lg.make_batch_step(config, protocol.training_samples),
        network, spec, data, target_variance, protocol,
    )
    show_result(
        "NORMAL vs exact(x,y)", normal.network, normal.status, normal.epochs,
        exact_xy, spec, data, protocol,
    )

    print("\nStarting true-z BYPASS training...", flush=True)
    network, _ = lg._build_network(config, spec, protocol)
    batches = dtl._bypass_batches(x, z, y, target_variance, protocol)
    network, status, epochs = dtl.train_bypass(network, config, batches, protocol)
    show_result(
        "BYPASS vs exact(x,z)", network, status, epochs,
        exact_xz, spec, data, protocol,
    )


if __name__ == "__main__":
    run()
