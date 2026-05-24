"""MNIST experiment orchestrator for the Distributional BPCN.

Single CLI entry point for class-incremental MNIST experiments. The
argparse surface IS the user-facing tunable surface: every BPCN
hyperparameter from `distributional_predictive_coding_v2.pdf` (network
shape, learning rates, priors, residual variance, E-step iterations,
M-step gradient scaling, target encoding) is exposed.

Usage
-----
    python -m experiments.mnist --help
    python -m experiments.mnist --epochs 5 --classes 0,1 --run-dir runs/mnist

Dependency direction
--------------------
`experiments.*` imports from `bpcn.*` only; `bpcn` must never import from
`experiments` (one-way dependency enforced by grep in plan section 5).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from bpcn.configs.base import BaseConfig
from bpcn.data.mnist import count_batches, iter_minibatches, load_split
from bpcn.evaluation.diagnostics import EpochDiagnostics, variance_decomposition
from bpcn.evaluation.predict import evaluate_split
from bpcn.inference.e_step import e_step
from bpcn.models.network import init_network
from bpcn.training.loop import make_batch_step


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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="experiments.mnist",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="BPCN class-incremental MNIST experiment.",
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
        "--hidden-dim", type=int, default=None,
        help="Hidden latent dim d_1 (Eq. 22).")
    g_net.add_argument(
        "--hidden-init", type=str, default=None, choices=["xavier", "he"],
        help="Init scaling kappa_1 for hidden weights (Eq. 98).")
    g_net.add_argument(
        "--output-init", type=str, default=None, choices=["xavier", "he"],
        help="Init scaling kappa_y for output weights (Eq. 98).")
    g_net.add_argument(
        "--init-log-var", type=float, default=None,
        help="Initial tau = log sigma_0^2 (Eq. 99, Section 8.1).")

    g_prior = p.add_argument_group("priors and residual variance (A4, A9; Eqs. 24, 27)")
    g_prior.add_argument(
        "--alpha-hidden", type=float, default=None,
        help="Prior scale alpha_1 for hidden weights (Eq. 27).")
    g_prior.add_argument(
        "--alpha-output", type=float, default=None,
        help="Prior scale alpha_y for output weights (Eq. 27).")
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
        help="Initial latent variance v_init (Eq. 100).")

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
        "--gamma-warmup-epochs", type=int, default=None,
        help="Linear gamma annealing window (Section 4.5 allows annealing).")
    g_m.add_argument(
        "--m-step-iters", type=_positive_int, default=None,
        help="Inner M-step gradient updates per minibatch (Section 6.7 'one or a few').")

    g_t = p.add_argument_group("target encoding (assumption I3, Section 8.2 Option 2)")
    g_t.add_argument(
        "--target-var", type=float, default=None,
        help="Gaussian-logit target variance epsilon_y (Section 8.2 Option 2).")

    g_se = p.add_argument_group("shared-energy extension (M-SE; shared_energy_dbpcn_extension.pdf)")
    g_se.add_argument(
        "--objective", type=str, default=None, choices=["pc_free_energy", "shared_dpc"],
        help="E-step objective: 'pc_free_energy' = legacy Eq. 40; "
             "'shared_dpc' = extension Eq. 63 F_kappa (default).")
    g_se.add_argument(
        "--kappa-start", type=float, default=None,
        help="Initial kappa for the kappa-homotopy ramp (extension Eq. 63). "
             "Ignored when --kappa-warmup-epochs == 0.")
    g_se.add_argument(
        "--kappa-warmup-epochs", type=int, default=None,
        help="Linearly ramp kappa from --kappa-start to 1.0 over the first K epochs "
             "(extension Eq. 63). 0 = no homotopy (kappa = 1 from epoch 1).")
    g_se.add_argument(
        "--rho-z", type=float, default=None,
        help="Proximal latent damping coefficient (extension Eq. 65). 0 = off.")
    g_se.add_argument(
        "--rho-w", type=float, default=None,
        help="Proximal weight damping coefficient (extension Eq. 64). 0 = off.")
    g_se.add_argument(
        "--r-max", type=float, default=None,
        help="Bounded variance residual r_max (extension Eq. 67). None = off.")
    g_se.add_argument(
        "--accept-or-damp", action="store_true",
        help="Enable accept-or-damp M-step (extension Eq. 68).")
    g_se.add_argument(
        "--accept-damp-omega", type=float, default=None,
        help="Omega for accept-or-damp interpolation (extension Eq. 68).")
    g_se.add_argument(
        "--accept-damp-tol", type=float, default=None,
        help="Tolerance on F_DPC increase before damping is triggered.")

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
        help="Target-free test-time initial latent variance; defaults to --v-init.")
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
    "input_dim", "hidden_dim", "hidden_init", "output_init", "init_log_var",
    "alpha_hidden", "alpha_output", "beta_inv_hidden", "beta_inv_output",
    "T_z", "eta_m", "eta_u", "v_init",
    "eta_mu_hidden", "eta_tau_hidden", "eta_mu_output", "eta_tau_output",
    "gamma_hidden", "gamma_output", "gamma_warmup_epochs", "m_step_iters",
    "target_var",
    # M-SE shared-energy extension fields.
    "objective", "kappa_start", "kappa_warmup_epochs",
    "rho_z", "rho_w", "r_max",
    "accept_or_damp", "accept_damp_omega", "accept_damp_tol",
    "epochs", "eval_every", "seed", "mc_samples",
    "eval_T_z", "eval_eta_m", "eval_eta_u", "eval_v_init",
    "run_dir",
)


def build_config(args) -> BaseConfig:
    """Build a BaseConfig by overlaying CLI overrides on the BaseConfig defaults.

    Any CLI argument that the user left at its default (None or False) keeps
    the BaseConfig value. BaseConfig is frozen, so we use dataclasses.replace.
    `accept_or_damp` is a store_true flag, so override only when True.
    """
    cfg = BaseConfig()
    overrides = {}
    for f in _CFG_FIELDS:
        val = getattr(args, f, None)
        if val is None:
            continue
        # `accept_or_damp` is store_true; only override when explicitly set.
        if f == "accept_or_damp" and not val:
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
    log_fn=print,
) -> dict:
    """Pure-function training driver.

    Returns a dict with keys:
      - history: JSON-serialisable per-epoch list (matches runs/base/history.json layout).
      - net    : final Network pytree.
    """
    log_fn(f"[bpcn.mnist] Loading MNIST classes {cfg.classes} ...")
    train = load_split(cfg.classes, train=True, seed=cfg.seed,
                       n_train=n_train_load, n_test=n_test_load)
    test = load_split(cfg.classes, train=False, seed=cfg.seed,
                      n_train=n_train_load, n_test=n_test_load)
    N_train = len(train.x)
    log_fn(f"[bpcn.mnist]   train={N_train}, test={len(test.x)}")

    log_fn(f"[bpcn.mnist] Initializing network {cfg.layer_dims} ...")
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
    )

    batch_step = make_batch_step(cfg, N_train=N_train)
    history = []

    for epoch in range(1, cfg.epochs + 1):
        ep_diag = EpochDiagnostics()
        t0 = time.time()
        n_batches = count_batches(train, cfg.batch_size, drop_last=True)
        kappa_epoch = jnp.float32(cfg.kappa_for_epoch(epoch))
        for bi, batch in enumerate(iter_minibatches(
            train,
            batch_size=cfg.batch_size,
            n_classes=cfg.output_dim,
            target_var=cfg.target_var,
            shuffle_seed=cfg.seed + epoch,
        )):
            net, e_diag, m_diag, f_dpc = batch_step(
                net,
                jnp.asarray(batch.x),
                jnp.asarray(batch.y_mean),
                jnp.asarray(batch.y_var),
                kappa_epoch,
            )
            ep_diag.add(e_diag, m_diag, f_dpc=f_dpc, kappa=float(kappa_epoch))
            if (bi + 1) % 20 == 0 or bi == n_batches - 1:
                last = ep_diag.records[-1]
                log_fn(
                    f"[bpcn.mnist]   epoch {epoch} batch {bi+1}/{n_batches} "
                    f"F_init={last['F_initial']:.4f} F_final={last['F_final']:.4f} "
                    f"F_DPC={last['F_DPC_total']:.4f} kappa={last['kappa']:.2f} "
                    f"kl_data(out)={last['output/kl_data']:.4f}"
                )

        summary = ep_diag.summary()
        elapsed = time.time() - t0
        log_fn(
            f"[bpcn.mnist] epoch {epoch} done in {elapsed:.1f}s; "
            f"avg F_init={summary['F_initial']:.4f} F_final={summary['F_final']:.4f} "
            f"kl_data(hid)={summary['hidden/kl_data']:.4f} "
            f"kl_data(out)={summary['output/kl_data']:.4f}"
        )
        epoch_record = {"epoch": epoch, "elapsed_s": elapsed, "summary": summary}

        if run_eval and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            key, ek = jax.random.split(key)
            metrics = evaluate_split(net, test, cfg, ek, batch_size=256)
            # Variance decomposition uses the target-free test-time E-step (Section 6.6).
            xb = jnp.asarray(test.x[:256])
            yb = jnp.zeros((256, cfg.output_dim), dtype=xb.dtype)
            frozen, _ = e_step(
                net, xb, yb,
                T_z=cfg.eval_T_z_resolved,
                eta_m=cfg.eval_eta_m_resolved,
                eta_u=cfg.eval_eta_u_resolved,
                v_init=cfg.eval_v_init_resolved,
                output_weight=0.0,
            )
            vd = variance_decomposition(net, frozen.m_z, frozen.v_z, layer_idx=-1)
            log_fn(
                f"[bpcn.mnist] epoch {epoch} eval: acc={metrics['accuracy']:.4f} "
                f"NLL={-metrics['log_likelihood_mean']:.4f} "
                f"H_mean={metrics['entropy_mean']:.4f}"
            )
            log_fn(
                f"[bpcn.mnist] epoch {epoch} variance decomposition (output layer): "
                f"residual={vd['residual_frac']:.3f} "
                f"propagated={vd['propagated_frac']:.3f} "
                f"epistemic={vd['epistemic_frac']:.3f}"
            )
            epoch_record["metrics"] = metrics
            epoch_record["var_decomp"] = vd

        history.append(epoch_record)

    return {"history": history, "net": net}


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------

def save_artifacts(cfg: BaseConfig, result: dict) -> str:
    """Write config.json, history.json, weights.npz to cfg.run_dir.

    On-disk layout matches `runs/base/` so existing analyses keep working.
    """
    os.makedirs(cfg.run_dir, exist_ok=True)

    cfg_json = {k: list(v) if isinstance(v, tuple) else v
                for k, v in dataclasses.asdict(cfg).items()}
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(cfg_json, f, indent=2)

    with open(os.path.join(cfg.run_dir, "history.json"), "w") as f:
        json.dump(result["history"], f, indent=2)

    net = result["net"]
    weights = {}
    for li, layer in enumerate(net.layers):
        weights[f"layer_{li}_mu"] = np.asarray(layer.mu)
        weights[f"layer_{li}_tau"] = np.asarray(layer.tau)
    np.savez(os.path.join(cfg.run_dir, "weights.npz"), **weights)

    return cfg.run_dir


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
    out = save_artifacts(cfg, result)
    print(f"[bpcn.mnist] DONE. Artefacts in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
