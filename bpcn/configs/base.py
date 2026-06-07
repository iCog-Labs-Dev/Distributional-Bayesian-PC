"""BPCN configuration dataclass.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 6.1: modeling assumptions.
- Section 6.2: initialization scales (Eqs. 98-99).
- Section 6.7: stopping criteria (Eq. 106).

References (extension: shared_energy_dbpcn_extension.pdf):
- Eq. 12: F_DPC, the canonical shared free energy descended by training.
- Algorithm 3: training-loop hyperparameters.

References (continuation: categorical_output_dbpcn_continuation.pdf):
- Section 4.1/4.2: MEAN vs MC categorical estimator (output_estimator).
- Eq. 2/48: F_cat-DPC = lambda_y F_out + F_trans-DPC + F_weight-KL.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class BaseConfig:
    # --- Dataset (plan §3.1) -------------------------------------------------
    classes: Tuple[int, ...] = (0, 1)        # base = 2-class MNIST (digits 0, 1)
    batch_size: int = 128
    # n_train_total / n_test_total are populated at runtime by
    # experiments.mnist.run(...) AFTER class filtering: they hold the actual
    # number of (post-filter) train / test examples used by this run, so the
    # saved config.json reflects the real dataset size. Default 0 means
    # "not yet populated" -- before run() has loaded the data. NOT consumed by
    # any training math; informational only.
    n_train_total: int = 0
    n_test_total: int = 0

    # --- Architecture (v2 Eqs. 7, 20-22) -------------------------------------
    input_dim: int = 784
    # `hidden_dims[l]` is the width of hidden latent z^{l+1} (zero-indexed in
    # code, one-indexed in the v2 write-up). For a single hidden layer this is
    # `(128,)`; for deeper nets (e.g. `(256, 128)`) the layer_dims become
    # `(input, 256, 128, output)`.
    hidden_dims: Tuple[int, ...] = (128,)
    # Per-layer feature maps psi_l (v2 Eq. 21, Section 4.5). Length must
    # equal len(hidden_dims). Each element is one of
    # {"identity", "relu", "leaky_relu", "tanh"}.
    activations: Tuple[str, ...] = ("identity",)

    # --- Initialization (Eqs. 98-99, plan U5) --------------------------------
    init_log_var: float = -6.0               # tau_0 = log sigma_0^2  (sigma_0 ~ 0.05)
    hidden_init: str = "xavier"
    output_init: str = "xavier"

    # --- Priors and residual variance (Eqs. 24, 27, A4, plan U5) -------------
    alpha_hidden: float = 1.0                # prior scale for hidden weights
    alpha_output: float = 1.0
    beta_inv_hidden: float = 1e-2            # residual variance B_l^{-1}  (fixed)
    beta_inv_output: float = 1e-2

    # --- E-step (Algorithm 1, plan U2-U3) ------------------------------------
    T_z: int = 8                             # inner E-step iterations
    eta_m: float = 0.1                       # latent mean learning rate
    eta_u: float = 0.05                      # latent log-var learning rate
    v_init: float = 1e-2                     # minimum initial latent variance floor

    # --- M-step (Algorithm 2, Eqs. 81-82, plan U4-U5) ------------------------
    eta_mu_hidden: float = 1e-3              # hidden mean learning rate
    eta_tau_hidden: float = 1e-4             # hidden log-var learning rate
    eta_mu_output: float = 1e-3
    eta_tau_output: float = 1e-4
    gamma_hidden: float = 1.0                # prior KL coefficient gamma_l (Eq. 62)
    gamma_output: float = 1.0
    m_step_iters: int = 1                    # inner M-step gradient updates per batch
                                             # (Section 6.7: "one or a few" updates)

    # --- Output target (assumption I3) ---------------------------------------
    # NOTE: `target_var` and `target_scale` are consumed only when
    # `output_likelihood == "gaussian"` (the I3 Gaussian-logit head). Under
    # `output_likelihood == "categorical"` (categorical_output_dbpcn_continuation.pdf)
    # they are ignored; the loss is softmax NLL and the target is the integer
    # class index `y_idx`.
    target_var: float = 1e-3                 # epsilon_y for Gaussian one-hot targets
    target_scale: float = 1.0                # one-hot peak magnitude in logit space.
                                             # 1.0 caps softmax confidence (~0.23 for
                                             # C=10) even at perfect regression; raise
                                             # (e.g. ~5) to lower the NLL/entropy floor.

    # --- Categorical output head (continuation note) -------------------------
    # References: categorical_output_dbpcn_continuation.pdf.
    # `output_likelihood`:
    #   "gaussian"    (default, legacy) -- the I3 Gaussian-logit head; the
    #                 output term in F is the inclusion-KL between
    #                 N(y_mean, y_var) and the Gaussian predictive (m_p^L,
    #                 v_p^L). `target_var`/`target_scale` control encoding.
    #   "categorical" -- Bayesian softmax head p(y=c|z^L,W_y) = softmax(W_y
    #                 psi_L(z^L))_c (Eq. 1/18 of the continuation). Output
    #                 term in F is the categorical NLL averaged over the
    #                 chosen estimator. Target is `y_idx` (integer class).
    # `output_estimator` selects the F_out estimator when categorical:
    #   "mean"        -- posterior-mean classifier (Eq. 21–22 of continuation).
    #                 Deterministic; data gradient updates only mu_y / m_z^L.
    #   "mc"          -- MC categorical NLL via reparameterized weights and
    #                 latents (Eq. 24–27). Bayes-by-Backprop-style gradient
    #                 on (mu_y, tau_y, m_z^L, v_z^L).
    # `mc_samples_train` is S used by the MC estimator during training (1 is
    # the BBB default). `lambda_y` is the output-loss weight in Eq. 2/48 of
    # the continuation (F_cat-DPC = lambda_y F_out + F_trans-DPC + F_weight-KL);
    # 1.0 = matched to F_trans-DPC.
    output_likelihood: str = "gaussian"
    output_estimator: str = "mean"
    mc_samples_train: int = 1
    lambda_y: float = 1.0

    # --- Training (Algorithm 3) ----------------------------------------------
    epochs: int = 5
    eval_every: int = 1                      # evaluate every K epochs
    seed: int = 0

    # --- Evaluation (Section 6.6) --------------------------------------------
    mc_samples: int = 16                     # S in Eq. 103
    eval_T_z: Optional[int] = None           # target-free test-time E-step; None -> T_z
    eval_eta_m: Optional[float] = None       # None -> eta_m
    eval_eta_u: Optional[float] = None       # None -> eta_u
    eval_v_init: Optional[float] = None      # target-free variance floor; None -> v_init

    # --- Output directory ----------------------------------------------------
    run_dir: str = "runs/base"

    def __post_init__(self):
        if self.m_step_iters < 1:
            raise ValueError(f"m_step_iters must be >= 1, got {self.m_step_iters}")
        if self.T_z < 1:
            raise ValueError(f"T_z must be >= 1, got {self.T_z}")
        if self.eval_T_z is not None and self.eval_T_z < 1:
            raise ValueError(f"eval_T_z must be >= 1, got {self.eval_T_z}")
        for name in ("eta_m", "eta_u", "v_init"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("eval_eta_m", "eval_eta_u", "eval_v_init"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                raise ValueError(f"{name} must be > 0 when set, got {val}")
        # Tuples loaded from JSON arrive as lists; coerce to tuple for
        # downstream consistency (so e.g. `cfg.hidden_dims[0]` and
        # `tuple(cfg.hidden_dims)` are identical). Frozen-dataclass pattern:
        # `object.__setattr__` is required.
        if isinstance(self.hidden_dims, list):
            object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if isinstance(self.activations, list):
            object.__setattr__(self, "activations", tuple(self.activations))

        # Validate the canonical multi-layer architecture fields.
        _ALLOWED_PSI = ("identity", "relu", "leaky_relu", "tanh")
        if not self.hidden_dims:
            raise ValueError("hidden_dims must be a non-empty tuple of layer widths")
        if not self.activations:
            raise ValueError("activations must be a non-empty tuple of feature maps")
        if len(self.activations) != len(self.hidden_dims):
            raise ValueError(
                f"activations length ({len(self.activations)}) must equal "
                f"hidden_dims length ({len(self.hidden_dims)}); got "
                f"activations={self.activations}, hidden_dims={self.hidden_dims}"
            )
        for a in self.activations:
            if a not in _ALLOWED_PSI:
                raise ValueError(
                    f"each activations entry must be one of {_ALLOWED_PSI}, "
                    f"got {a!r} in {self.activations}"
                )
        for d in self.hidden_dims:
            if d < 1:
                raise ValueError(
                    f"each hidden_dims entry must be >= 1, got {d} in {self.hidden_dims}"
                )
        if self.target_scale <= 0:
            raise ValueError(f"target_scale must be > 0, got {self.target_scale}")
        if self.output_likelihood not in ("gaussian", "categorical"):
            raise ValueError(
                f"output_likelihood must be 'gaussian' or 'categorical', "
                f"got {self.output_likelihood!r}"
            )
        if self.output_estimator not in ("mean", "mc"):
            raise ValueError(
                f"output_estimator must be 'mean' or 'mc', got {self.output_estimator!r}"
            )
        if self.mc_samples_train < 1:
            raise ValueError(f"mc_samples_train must be >= 1, got {self.mc_samples_train}")
        if self.lambda_y <= 0:
            raise ValueError(f"lambda_y must be > 0, got {self.lambda_y}")

    @property
    def output_dim(self) -> int:
        return len(self.classes)

    @property
    def layer_dims(self) -> Tuple[int, ...]:
        """Full layer-dim tuple `(input_dim, *hidden_dims, output_dim)`."""
        return (self.input_dim, *self.hidden_dims, self.output_dim)

    @property
    def L_hidden(self) -> int:
        return len(self.hidden_dims)

    @property
    def eval_T_z_resolved(self) -> int:
        return self.T_z if self.eval_T_z is None else self.eval_T_z

    @property
    def eval_eta_m_resolved(self) -> float:
        return self.eta_m if self.eval_eta_m is None else self.eval_eta_m

    @property
    def eval_eta_u_resolved(self) -> float:
        return self.eta_u if self.eval_eta_u is None else self.eval_eta_u

    @property
    def eval_v_init_resolved(self) -> float:
        return self.v_init if self.eval_v_init is None else self.eval_v_init
