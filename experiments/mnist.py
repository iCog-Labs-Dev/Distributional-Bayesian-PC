"""MNIST experiment orchestrator for the Distributional BPCN.

Single CLI entry point for MNIST experiments. The argparse surface IS the
user-facing tunable surface: every BPCN hyperparameter from
`distributional_predictive_coding_v2.pdf` (network shape, learning rates,
priors, residual variance, E-step iterations, M-step gradient scaling,
target encoding) is exposed, plus the categorical-output controls from
`categorical_output_dbpcn_continuation.pdf`.

Usage
-----
    python -m experiments.mnist --help
    python -m experiments.mnist --epochs 5 --classes 0,1 --run-dir runs/mnist
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from sys import path
path.append(".")

import jax
import jax.numpy as jnp
import numpy as np

from bpcn.configs.base import BaseConfig
from bpcn.data.mnist import count_batches, iter_minibatches, load_split
from bpcn.evaluation.diagnostics import EpochDiagnostics, variance_decomposition
from bpcn.evaluation.predict import evaluate_split
from bpcn.inference.e_step import e_step
from bpcn.inference.feature_moments import psi_moments
from bpcn.models.network import init_network
from bpcn.training.loop import make_batch_step
from logger import get_logger

_log = get_logger("mnist")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_classes(s: str):
    """Parse '0,1,2' -> (0, 1, 2)."""
    parts = [c.strip() for c in s.split(",") if c.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--classes requires at least one digit")
    return tuple(int(c) for c in parts)


def _positive_int(s: str) -> int:
    """Argparse type for counts that must be at least one."""
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return v


def _positive_float(s: str) -> float:
    """Argparse type for positive scalar hyperparameters."""
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return v


def _parse_int_list(s: str):
    """Parse '256,128,64' -> (256, 128, 64). Used for --hidden-dims."""
    parts = [c.strip() for c in s.split(",") if c.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("comma-separated list must be non-empty")
    out = []
    for p in parts:
        try:
            v = int(p)
        except ValueError:
            raise argparse.ArgumentTypeError(f"could not parse {p!r} as int")
        if v < 1:
            raise argparse.ArgumentTypeError(f"each entry must be >= 1, got {v}")
        out.append(v)
    return tuple(out)


_ALLOWED_ACTIVATIONS = ("identity", "relu", "leaky_relu", "tanh")


def _parse_activations(s: str):
    """Parse 'relu,tanh' -> ('relu', 'tanh'). Used for --activations."""
    parts = [c.strip() for c in s.split(",") if c.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--activations must be non-empty")
    for p in parts:
        if p not in _ALLOWED_ACTIVATIONS:
            raise argparse.ArgumentTypeError(
                f"each activation must be one of {_ALLOWED_ACTIVATIONS}, got {p!r}"
            )
    return tuple(parts)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="experiments.mnist",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="BPCN MNIST experiment.",
    )

    g_data = p.add_argument_group("dataset")
    g_data.add_argument(
        "--classes", type=_parse_classes, default=None,
        help="Comma-separated MNIST digit indices (Section 3.1 Eq. 20). Example: '0,1'.")
    g_data.add_argument(
        "--batch-size", type=int, default=None,
        help="Minibatch size B used by Algorithm 3.")
    g_data.add_argument(
        "--n-train", type=int, default=60000,
        help="Max MNIST train examples loaded before class filter.")
    g_data.add_argument(
        "--n-test", type=int, default=10000,
        help="Max MNIST test examples loaded before class filter.")

    g_net = p.add_argument_group("network")
    g_net.add_argument(
        "--input-dim", type=int, default=None,
        help="Input dim d_0 (Eq. 20; 784 for flattened MNIST).")
    g_net.add_argument(
        "--hidden-dims", type=_parse_int_list, default=None,
        help="Comma-separated hidden layer dims for multi-layer DBPCN. "
             "E.g. '256,128' builds L_hidden=2 with widths d_1=256, d_2=128. "
             "Length must equal --activations length.")
    g_net.add_argument(
        "--hidden-init", type=str, default=None, choices=["xavier", "he"],
        help="Init scaling kappa_l for hidden weights (Eq. 98). Shared across "
             "all hidden layers.")
    g_net.add_argument(
        "--output-init", type=str, default=None, choices=["xavier", "he"],
        help="Init scaling kappa_y for output weights (Eq. 98).")
    g_net.add_argument(
        "--init-log-var", type=float, default=None,
        help="Initial tau = log sigma_0^2 (Eq. 99, Section 8.1).")
    g_net.add_argument(
        "--activations", type=_parse_activations, default=None,
        help="Comma-separated per-layer activations psi_l. Length must equal "
             "L_hidden (= len(--hidden-dims)). Each entry is one of "
             "{identity, relu, leaky_relu, tanh}. E.g. 'relu,tanh' for L=2.")

    g_prior = p.add_argument_group("priors and residual variance (A4, A9; Eqs. 24, 27)")
    g_prior.add_argument(
        "--alpha-hidden", type=float, default=None,
        help="Prior scale alpha_1 for hidden weights (Eq. 27).")
    g_prior.add_argument(
        "--alpha-output", type=float, default=None,
        help="Prior scale alpha_y for output weights (Eq. 27).")
    g_prior.add_argument(
        "--alpha-scheme", type=str, default=None,
        choices=["constant", "matched_he", "matched_xavier"],
        help="Layerwise weight-prior scaling scheme (DBPCN/dbpcn_weight_kl_sigma2_continuation.pdf §5). "
             "'constant' (default) uses --alpha-hidden / --alpha-output / --init-log-var as passed. "
             "'matched_he' sets per-layer α²_l = σ²_w,0,l = 2/fan_in_l (ReLU networks). "
             "'matched_xavier' sets α²_l = σ²_w,0,l = 1/fan_in_l (tanh/identity). "
             "Under matched_*, --alpha-hidden / --alpha-output / --init-log-var are IGNORED.")
    g_prior.add_argument(
        "--beta-inv-hidden", type=float, default=None,
        help="Residual variance beta_1^{-1} (Eq. 24, fixed per A4).")
    g_prior.add_argument(
        "--beta-inv-output", type=float, default=None,
        help="Residual variance beta_y^{-1} (Eq. 24, fixed per A4).")

    g_e = p.add_argument_group("E-step (Algorithm 1)")
    g_e.add_argument(
        "--t-z", dest="T_z", type=int, default=None,
        help="Number of inner E-step iterations T_z (Section 6.3 Algorithm 1).")
    g_e.add_argument(
        "--eta-m", type=float, default=None,
        help="Latent mean learning rate (Eq. 43).")
    g_e.add_argument(
        "--eta-u", type=float, default=None,
        help="Latent log-variance learning rate (Eq. 43).")
    g_e.add_argument(
        "--v-init", type=float, default=None,
        help="Minimum latent variance floor used by predictive latent initialization.")
    g_e.add_argument(
        "--init-perturb-std", type=float, default=None,
        help="Predictive-disequilibrium init coefficient c_m "
             "Adds natural-scale mean jitter c_m * sqrt(v_pred) * xi to each "
             "hidden latent at the start of every training E-step. 0.0 (default) "
             "= exact predictive init (v2 Eq. 89). Training-only; the seed-free "
             "target-free eval always uses 0.0.")
    g_e.add_argument(
        "--init-log-var-offset", type=float, default=None,
        help="Additive offset applied to the per-HIDDEN-layer init τ AFTER the "
             "--alpha-scheme derivation (or to --init-log-var under "
             "alpha_scheme=constant). 0.0 (default) preserves byte-identical "
             "init. Reference: DBPCN/dbpcn_weight_kl_sigma2_continuation.pdf "
             "§5 step 3 — raises σ²₀ so the data-side τ-gradient Δτ ∝ "
             "r·σ²·H²/(2v_p²) is not 400× weaker than the prior pull. E.g. "
             "+2.0 with --alpha-scheme matched_he at fan_in=784 takes τ₀ from "
             "log(2/784)≈-5.97 to ≈-3.97 (σ²₀ from 0.0026 to 0.019). Output "
             "layer init τ is intentionally NOT offset.")

    g_m = p.add_argument_group("M-step (Algorithm 2; Eqs. 81-82)")
    g_m.add_argument(
        "--eta-mu-hidden", type=float, default=None,
        help="Learning rate for hidden mu (Eq. 81).")
    g_m.add_argument(
        "--eta-tau-hidden", type=float, default=None,
        help="Learning rate for hidden tau (Eq. 82).")
    g_m.add_argument(
        "--eta-mu-output", type=float, default=None,
        help="Learning rate for output mu (Eq. 81).")
    g_m.add_argument(
        "--eta-tau-output", type=float, default=None,
        help="Learning rate for output tau (Eq. 82).")
    g_m.add_argument(
        "--gamma-hidden", type=float, default=None,
        help="Prior KL coefficient gamma_1 in Eq. 62.")
    g_m.add_argument(
        "--gamma-output", type=float, default=None,
        help="Prior KL coefficient gamma_y in Eq. 62.")
    g_m.add_argument(
        "--gamma-mu-hidden", type=_positive_float, default=None,
        help="Decoupled μ-prior coefficient at hidden layers "
             "When set, the μ-side prior gradient (Eq. 81 second term) uses "
             "this γ_μ value instead of --gamma-hidden; the τ-side (Eq. 82) "
             "is unchanged. Useful with --alpha-scheme matched_he, where the "
             "matched-narrow α² otherwise multiplies the μ-prior pull beyond "
             "intended. Default None = legacy single-γ behaviour.")
    g_m.add_argument(
        "--gamma-mu-output", type=_positive_float, default=None,
        help="Same as --gamma-mu-hidden but for the output layer's μ-prior.")
    g_m.add_argument(
        "--m-step-iters", type=_positive_int, default=None,
        help="Inner M-step gradient updates per minibatch (Section 6.7 'one or a few').")

    g_t = p.add_argument_group("target encoding (assumption I3, Section 8.2 Option 2)")
    g_t.add_argument(
        "--target-var", type=float, default=None,
        help="Gaussian-logit target variance epsilon_y (Section 8.2 Option 2). "
             "Ignored under --output-likelihood categorical.")
    g_t.add_argument(
        "--target-scale", type=_positive_float, default=None,
        help="One-hot peak magnitude in logit space (default 1.0). Raise (e.g. 5) "
             "to lower the softmax confidence ceiling and the NLL/entropy floor. "
             "Ignored under --output-likelihood categorical.")

    g_cat = p.add_argument_group(
        "categorical output head"
    )
    g_cat.add_argument(
        "--output-likelihood", type=str, default=None,
        choices=["gaussian", "categorical"],
        help="Output boundary likelihood. 'gaussian' = legacy I3 Gaussian-logit "
             "head (default). 'categorical' = Bayesian softmax head "
             "p(y=c|z^L,W_y) = softmax(W_y psi_L(z^L))_c (Eq. 1/18 of the "
             "continuation note).")
    g_cat.add_argument(
        "--output-estimator", type=str, default=None,
        choices=["mean", "mc"],
        help="F_out estimator for the categorical head. 'mean' = posterior-mean "
             "classifier (Eq. 21). 'mc' = reparameterized Monte Carlo (Eq. 27). "
             "Ignored under --output-likelihood gaussian.")
    g_cat.add_argument(
        "--mc-samples-train", type=_positive_int, default=None,
        help="S in Eq. 27 for the MC categorical training estimator. 1 is the "
             "BBB default; larger reduces gradient variance. Ignored under "
             "--output-estimator mean.")
    g_cat.add_argument(
        "--lambda-y", type=_positive_float, default=None,
        help="Output-loss weight lambda_y in F_cat-DPC = lambda_y F_out + "
             "F_trans-DPC + F_weight-KL. 1.0 matches "
             "the hidden transition scale.")

    g_train = p.add_argument_group("training and evaluation")
    g_train.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs in Algorithm 3.")
    g_train.add_argument(
        "--eval-every", type=int, default=None,
        help="Eval cadence in epochs.")
    g_train.add_argument(
        "--seed", type=int, default=None,
        help="PRNGKey root seed.")
    g_train.add_argument(
        "--mc-samples", type=int, default=None,
        help="S in Eq. 103 (Monte Carlo predictive samples at eval).")
    g_train.add_argument(
        "--eval-t-z", dest="eval_T_z", type=_positive_int, default=None,
        help="Target-free test-time E-step iterations; defaults to --t-z (Section 6.6).")
    g_train.add_argument(
        "--eval-eta-m", type=_positive_float, default=None,
        help="Target-free test-time latent mean learning rate; defaults to --eta-m.")
    g_train.add_argument(
        "--eval-eta-u", type=_positive_float, default=None,
        help="Target-free test-time latent log-variance learning rate; defaults to --eta-u.")
    g_train.add_argument(
        "--eval-v-init", type=_positive_float, default=None,
        help="Target-free test-time latent variance floor; defaults to --v-init.")
    g_train.add_argument(
        "--no-eval", action="store_true",
        help="Skip MC eval each epoch (faster smoke runs).")

    g_out = p.add_argument_group("output")
    g_out.add_argument(
        "--run-dir", type=str, default=None,
        help="Directory for config.json / history.json / weights.npz artefacts.")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Build BaseConfig from CLI overrides
# ---------------------------------------------------------------------------

# CLI dest -> BaseConfig field. CLI flags use kebab-case which argparse maps
# to snake_case dest; T_z is the one exception (preserves the write-up symbol).
_CFG_FIELDS = (
    "classes", "batch_size",
    "input_dim", "hidden_dims", "hidden_init", "output_init", "init_log_var",
    "init_log_var_offset",
    "activations",
    "alpha_hidden", "alpha_output", "alpha_scheme", "beta_inv_hidden", "beta_inv_output",
    "T_z", "eta_m", "eta_u", "v_init", "init_perturb_std",
    "eta_mu_hidden", "eta_tau_hidden", "eta_mu_output", "eta_tau_output",
    "gamma_hidden", "gamma_output", "gamma_mu_hidden", "gamma_mu_output", "m_step_iters",
    "target_var", "target_scale",
    # Categorical output head.
    "output_likelihood", "output_estimator", "mc_samples_train", "lambda_y",
    "epochs", "eval_every", "seed", "mc_samples",
    "eval_T_z", "eval_eta_m", "eval_eta_u", "eval_v_init",
    "run_dir",
)


def build_config(args) -> BaseConfig:
    """Build a BaseConfig by overlaying CLI overrides on the BaseConfig defaults.

    Any CLI argument that the user left at its default (None) keeps the
    BaseConfig value. BaseConfig is frozen, so we use dataclasses.replace.
    """
    cfg = BaseConfig()
    overrides = {}
    for f in _CFG_FIELDS:
        val = getattr(args, f, None)
        if val is None:
            continue
        overrides[f] = val
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return cfg


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

def run(
    cfg: BaseConfig,
    *,
    n_train_load: int = 60000,
    n_test_load: int = 10000,
    run_eval: bool = True,
) -> dict:
    """Pure-function training driver.

    Returns a dict with keys:
      - history: JSON-serialisable per-epoch list (matches runs/base/history.json layout).
      - net    : final Network pytree.
    """
    _log.info(f"Loading MNIST classes {cfg.classes} ...")
    train = load_split(cfg.classes, train=True, seed=cfg.seed,
                       n_train=n_train_load, n_test=n_test_load)
    test = load_split(cfg.classes, train=False, seed=cfg.seed,
                      n_train=n_train_load, n_test=n_test_load)
    N_train = len(train.x)
    N_test = len(test.x)
    _log.info(f"  train={N_train}, test={N_test}")
    cfg = dataclasses.replace(cfg, n_train_total=N_train, n_test_total=N_test)

    _log.info(f"Initializing network {cfg.layer_dims} ...")
    key = jax.random.PRNGKey(cfg.seed)
    key, init_key = jax.random.split(key)
    net = init_network(
        init_key,
        layer_dims=cfg.layer_dims,
        alpha_hidden=cfg.alpha_hidden,
        alpha_output=cfg.alpha_output,
        beta_inv_hidden=cfg.beta_inv_hidden,
        beta_inv_output=cfg.beta_inv_output,
        init_log_var=cfg.init_log_var,
        hidden_init=cfg.hidden_init,
        output_init=cfg.output_init,
        activations=cfg.activations,
        output_likelihood=cfg.output_likelihood,
        output_estimator=cfg.output_estimator,
        alpha_scheme=cfg.alpha_scheme,
        init_log_var_offset=cfg.init_log_var_offset,
    )
    if cfg.alpha_scheme != "constant":
        # Surface per-layer α / σ²_w,0 derived from fan_in so the matched-prior
        # scheme is loud in the run output.
        per_layer_alpha = [float(lr.alpha) for lr in net.layers]
        per_layer_sigma2 = [float(lr.tau[0, 0]) for lr in net.layers]
        _log.info(
            f"  alpha_scheme={cfg.alpha_scheme!r}: per-layer α "
            f"= {[round(a, 5) for a in per_layer_alpha]}, "
            f"init τ = {[round(t, 4) for t in per_layer_sigma2]} "
            f"(α_hidden/output and init_log_var fields IGNORED under matched scheme)"
        )

    batch_step = make_batch_step(cfg, N_train=N_train)
    history = []
    # Base key for the training-loop randomness consumed by the MC categorical output.
    train_base_key = jax.random.PRNGKey(cfg.seed + 1)

    # Initial (pre-training) F_weight_kl in per-data-point scale.
    from bpcn.inference.shared_energy import _weight_kl_total_decomposed
    weight_kl_initial = float(_weight_kl_total_decomposed(
        net, gamma_hidden=cfg.gamma_hidden, gamma_output=cfg.gamma_output,
        gamma_mu_hidden=cfg.gamma_mu_hidden,
        gamma_mu_output=cfg.gamma_mu_output,
        weight_kl_scale=1.0 / float(N_train),
    )[0])
    _log.info(
        f"  F_weight_kl baseline at init = {weight_kl_initial:.4f} nats/batch "
        f"(per-data-point scale)"
    )

    for epoch in range(1, cfg.epochs + 1):
        ep_diag = EpochDiagnostics()
        t0 = time.time()
        n_batches = count_batches(train, cfg.batch_size, drop_last=True)
        epoch_key = jax.random.fold_in(train_base_key, epoch)
        for bi, batch in enumerate(iter_minibatches(
            train,
            batch_size=cfg.batch_size,
            n_classes=cfg.output_dim,
            target_var=cfg.target_var,
            target_scale=cfg.target_scale,
            shuffle_seed=cfg.seed + epoch,
        )):
            batch_key = jax.random.fold_in(epoch_key, bi)
            net, e_diag, m_diag, f_dpc, init_res, freeze_res = batch_step(
                net,
                jnp.asarray(batch.x),
                jnp.asarray(batch.y_mean),
                jnp.asarray(batch.y_var),
                jnp.asarray(batch.y_idx),
                batch_key,
            )
            ep_diag.add(
                e_diag, m_diag, f_dpc=f_dpc,
                init_residuals=init_res, freeze_residuals=freeze_res,
                weight_kl_initial=weight_kl_initial,
            )
            if (bi + 1) % 20 == 0 or bi == n_batches - 1:
                last = ep_diag.records[-1]
                _log.info(
                    f"  epoch {epoch} batch {bi+1}/{n_batches} "
                    f"F_init={last['F_initial']:.4f} F_final={last['F_final']:.4f} "
                    f"F_DPC={last['F_DPC_total']:.4f} "
                    f"kl_data(out)={last['output/kl_data']:.4f}"
                )

        summary = ep_diag.summary()
        elapsed = time.time() - t0
        top_hidden = f"hidden_{cfg.L_hidden - 1}"
        _log.info(
            f"epoch {epoch} done in {elapsed:.1f}s; "
            f"avg F_init={summary['F_initial']:.4f} F_final={summary['F_final']:.4f} "
            f"kl_data({top_hidden})={summary[f'{top_hidden}/kl_data']:.4f} "
            f"kl_data(out)={summary['output/kl_data']:.4f} "
            f"init/K({top_hidden})={summary.get(f'{top_hidden}/init/K', 0.0):.3e} "
            f"freeze/K({top_hidden})={summary.get(f'{top_hidden}/freeze/K', 0.0):.3e}"
        )
        # Decomposed weight-KL diagnostic (continuation note §5 step 2).
        _log.info(
            f"epoch {epoch} F_weight_kl="
            f"{summary.get('F_weight_kl', 0.0):.4f}  "
            f"(μ={summary.get('F_weight_kl_mu', 0.0):.4f}  "
            f"var={summary.get('F_weight_kl_var', 0.0):.4f}  "
            f"Δ={summary.get('F_weight_kl_delta', 0.0):+.4f})  "
            f"F_DPC={summary.get('F_DPC_total', 0.0):.4f}"
        )
        epoch_record = {"epoch": epoch, "elapsed_s": elapsed, "summary": summary}

        if run_eval and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            key, ek = jax.random.split(key)
            metrics = evaluate_split(net, test, cfg, ek)
            # Variance decomposition uses the target-free test-time E-step
            # (descends F_DPC at output_weight=0) so the diagnostic is at the
            # same fixed point evaluate_split uses.
            xb = jnp.asarray(test.x[:256])
            yb = jnp.zeros((256, cfg.output_dim), dtype=xb.dtype)
            yvb = jnp.zeros_like(yb)
            frozen, _ = e_step(
                net, xb, yb,
                T_z=cfg.eval_T_z_resolved,
                eta_m=cfg.eval_eta_m_resolved,
                eta_u=cfg.eval_eta_u_resolved,
                v_init=cfg.eval_v_init_resolved,
                output_weight=0.0,
                y_var=yvb,
                gamma_hidden=cfg.gamma_hidden,
                gamma_output=cfg.gamma_output,
            )
            # Variance decomposition of the OUTPUT layer's predictive uses
            # the post-psi_L presynaptic moments.
            m_h_vd, v_h_vd = psi_moments(cfg.activations[-1], frozen.m_z, frozen.v_z)
            vd = variance_decomposition(net, m_h_vd, v_h_vd, layer_idx=-1)
            _log.info(
                f"epoch {epoch} eval: "
                f"MC[acc={metrics['accuracy']:.4f} "
                f"NLL={-metrics['log_likelihood_mean']:.4f} "
                f"H={metrics['entropy_mean']:.4f}]  "
                f"MEAN[acc={metrics['mean_accuracy']:.4f} "
                f"NLL={-metrics['mean_log_likelihood_mean']:.4f} "
                f"H={metrics['mean_entropy_mean']:.4f}]"
            )
            _log.info(
                f"epoch {epoch} variance decomposition (output layer): "
                f"residual={vd['residual_frac']:.3f} "
                f"propagated={vd['propagated_frac']:.3f} "
                f"epistemic={vd['epistemic_frac']:.3f}"
            )
            epoch_record["metrics"] = metrics
            epoch_record["var_decomp"] = vd

        history.append(epoch_record)

    return {"history": history, "net": net, "cfg": cfg}


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def _atomic_json_dump(path: str, obj) -> None:
    """Write JSON via os.replace so interrupted checkpoints keep prior files."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def _weights_dict(net) -> dict:
    weights = {}
    for li, layer in enumerate(net.layers):
        weights[f"layer_{li}_mu"] = np.asarray(layer.mu)
        weights[f"layer_{li}_tau"] = np.asarray(layer.tau)
    return weights


def _save_weights_npz(path: str, net) -> None:
    tmp_path = f"{path}.tmp.npz"
    np.savez(tmp_path, **_weights_dict(net))
    os.replace(tmp_path, path)


def _config_json(cfg: BaseConfig) -> dict:
    return {k: list(v) if isinstance(v, tuple) else v
            for k, v in dataclasses.asdict(cfg).items()}


def save_artifacts(cfg: BaseConfig, result: dict) -> str:
    """Write config.json, history.json, weights.npz to cfg.run_dir.

    On-disk layout matches `runs/base/` so existing analyses keep working.
    """
    os.makedirs(cfg.run_dir, exist_ok=True)

    _atomic_json_dump(os.path.join(cfg.run_dir, "config.json"), _config_json(cfg))
    _atomic_json_dump(os.path.join(cfg.run_dir, "history.json"), result["history"])
    _save_weights_npz(os.path.join(cfg.run_dir, "weights.npz"), result["net"])

    return cfg.run_dir


def _load_checkpoint(cfg: BaseConfig, checkpoint_dir: str):
    """Load history and weights from a previous run directory."""
    history_path = os.path.join(checkpoint_dir, "history.json")
    weights_path = os.path.join(checkpoint_dir, "weights.npz")
    with open(history_path) as f:
        history = json.load(f)

    net = init_network(
        jax.random.PRNGKey(cfg.seed),
        layer_dims=cfg.layer_dims,
        alpha_hidden=cfg.alpha_hidden,
        alpha_output=cfg.alpha_output,
        beta_inv_hidden=cfg.beta_inv_hidden,
        beta_inv_output=cfg.beta_inv_output,
        init_log_var=cfg.init_log_var,
        hidden_init=cfg.hidden_init,
        output_init=cfg.output_init,
        activations=cfg.activations,
        output_likelihood=cfg.output_likelihood,
        output_estimator=cfg.output_estimator,
        alpha_scheme=cfg.alpha_scheme,
        init_log_var_offset=cfg.init_log_var_offset,
    )

    weights = np.load(weights_path)
    layers = []
    for li, layer in enumerate(net.layers):
        mu_key = f"layer_{li}_mu"
        tau_key = f"layer_{li}_tau"
        if mu_key not in weights or tau_key not in weights:
            raise ValueError(f"checkpoint is missing {mu_key}/{tau_key}")
        mu = jnp.asarray(weights[mu_key])
        tau = jnp.asarray(weights[tau_key])
        if mu.shape != layer.mu.shape or tau.shape != layer.tau.shape:
            raise ValueError(
                f"checkpoint layer {li} has shape mu={mu.shape}, tau={tau.shape}; "
                f"expected mu={layer.mu.shape}, tau={layer.tau.shape}"
            )
        layers.append(layer._replace(mu=mu, tau=tau))
    weights.close()
    return history, net._replace(layers=tuple(layers))


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    result = run(
        cfg,
        n_train_load=args.n_train,
        n_test_load=args.n_test,
        run_eval=not args.no_eval,
    )
    cfg_runtime = result.get("cfg", cfg)
    out = save_artifacts(cfg_runtime, result)
    _log.info(f"DONE. Artefacts in {out}")
    return 0


if __name__ == "__main__":
    _log.info(f"Starting Distributional Bayesian Predictive Coding experiment on: {jax.default_backend()}")
    sys.exit(main())
