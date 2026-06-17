"""BPCN configuration dataclass."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class BaseConfig:

    classes: Tuple[int, ...] = (0, 1)        # base = 2-class MNIST (digits 0, 1)
    batch_size: int = 128
    n_train_total: int = 0
    n_test_total: int = 0
    input_dim: int = 784
    hidden_dims: Tuple[int, ...] = (128,)
    activations: Tuple[str, ...] = ("relu",)
    init_log_var: float = -6.0               # tau_0 = log sigma_0^2  (sigma_0 ~ 0.05)
    init_log_var_offset: float = 0.0
    hidden_init: str = "xavier"
    output_init: str = "xavier"
    alpha_hidden: float = 1.0                # prior scale for hidden weights
    alpha_output: float = 1.0
    beta_inv_hidden: float = 1e-2            # residual variance B_l^{-1}  (fixed)
    beta_inv_output: float = 1e-2
    alpha_scheme: str = "matched_he"
    T_z: int = 20                            # inner E-step iterations (§23 Stage 2)
    eta_m: float = 0.1                       # latent mean learning rate
    eta_u: float = 0.05                      # latent log-var learning rate
    v_init: float = 1e-2                     # minimum initial latent variance floor
    init_perturb_std: float = 0.1          # stddev of N(0, σ²) noise added to initial latent means (Section 6.7)
    eta_mu_hidden: float = 1e-2              # hidden mean learning rate
    eta_tau_hidden: float = 5e-2             # hidden log-var learning rate
    eta_mu_output: float = 1e-2
    eta_tau_output: float = 5e-3
    gamma_hidden: float = 0.1                # γ_τ under the γ-split (§23 Stage 2)
    gamma_output: float = 1.0
    gamma_mu_hidden: Optional[float] = 0.004  # Decoupled μ-prior coefficient 
    gamma_mu_output: Optional[float] = 0.004
    m_step_iters: int = 16                   # inner M-step gradient updates per batch
    target_var: float = 1e-3                 # epsilon_y for Gaussian one-hot targets
    target_scale: float = 1.0                # one-hot peak magnitude in logit space.
    output_likelihood: str = "categorical"
    output_estimator: str = "mc"
    mc_samples_train: int = 1
    lambda_y: float = 1.0

    # Training 
    epochs: int = 5
    eval_every: int = 1                      # evaluate every K epochs
    seed: int = 0

    # Evaluation
    mc_samples: int = 16                     # S in Eq. 103
    eval_T_z: Optional[int] = None           # target-free test-time E-step; None -> T_z
    eval_eta_m: Optional[float] = None       # None -> eta_m
    eval_eta_u: Optional[float] = None       # None -> eta_u
    eval_v_init: Optional[float] = None      # target-free variance floor; None -> v_init

    run_dir: str = "runs/base"        # directory for saving checkpoints, logs, etc.

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
        if self.init_perturb_std < 0:
            raise ValueError(
                f"init_perturb_std must be >= 0, got {self.init_perturb_std}"
            )
        _ALLOWED_ALPHA_SCHEMES = ("constant", "matched_he", "matched_xavier")
        if self.alpha_scheme not in _ALLOWED_ALPHA_SCHEMES:
            raise ValueError(
                f"alpha_scheme must be one of {_ALLOWED_ALPHA_SCHEMES}, "
                f"got {self.alpha_scheme!r}"
            )
        for name in ("gamma_mu_hidden", "gamma_mu_output"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                raise ValueError(f"{name} must be > 0 when set, got {val}")
        for name in ("eval_eta_m", "eval_eta_u", "eval_v_init"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                raise ValueError(f"{name} must be > 0 when set, got {val}")
            
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
