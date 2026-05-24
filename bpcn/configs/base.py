"""Base-stage configuration for 2-class MNIST BPCN.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 6.1: modeling assumptions for the simplest BPCN.
- Section 6.2: initialization scales (Eqs. 98-99).
- Section 6.7: stopping criteria (Eq. 106).

Constants here are implementation choices (plan §1.3 unknowns U2-U5).
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class BaseConfig:
    # --- Dataset (plan §3.1) -------------------------------------------------
    classes: Tuple[int, ...] = (0, 1)        # base = 2-class MNIST (digits 0, 1)
    batch_size: int = 128
    n_train_total: int = 12000               # approximate train count for digits 0, 1

    # --- Architecture (plan §3.1, I1, I2) ------------------------------------
    input_dim: int = 784
    hidden_dim: int = 128
    # output_dim derived from len(classes)

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
    v_init: float = 1e-2                     # initial latent variance (Eq. 100)

    # --- M-step (Algorithm 2, Eqs. 81-82, plan U4-U5) ------------------------
    eta_mu_hidden: float = 1e-3              # hidden mean learning rate
    eta_tau_hidden: float = 1e-4             # hidden log-var learning rate
    eta_mu_output: float = 1e-3
    eta_tau_output: float = 1e-4
    gamma_hidden: float = 1.0                # prior KL coefficient gamma_l (Eq. 62)
    gamma_output: float = 1.0
    gamma_warmup_epochs: int = 0             # linear warm-up; 0 disables
    m_step_iters: int = 1                    # inner M-step gradient updates per batch
                                             # (Section 6.7: "one or a few" updates)

    # --- Output target (assumption I3) ---------------------------------------
    target_var: float = 1e-3                 # epsilon_y for Gaussian one-hot targets

    # --- Shared-energy extension (M-SE) --------------------------------------
    # References: shared_energy_dbpcn_extension.pdf.
    # objective='pc_free_energy' (LEGACY) recovers runs/base/ numerics: the
    #   E-step descends Eq. 40 of the original BPCN write-up.
    # objective='shared_dpc'              switches the E-step to descend the
    #   shared energy F_kappa of extension Eq. 63 (which at kappa=1 is F_DPC
    #   of Eq. 12). The M-step gradient algebra is unchanged in either mode.
    objective: str = "shared_dpc"            # M-SE default: shared-energy E-step
    kappa_start: float = 1.0                 # initial kappa (extension Eq. 63)
    kappa_warmup_epochs: int = 0             # epochs to ramp kappa_start -> 1.0; 0 = no ramp
    rho_z: float = 0.0                       # proximal latent damping (Eq. 65); 0 = off
    rho_w: float = 0.0                       # proximal weight damping (Eq. 64); 0 = off
    r_max: Optional[float] = None            # bounded variance residual (Eq. 67); None = off
    accept_or_damp: bool = False             # accept-or-damp M-step (Eq. 68)
    accept_damp_omega: float = 0.5           # omega for damped update (Eq. 68)
    accept_damp_tol: float = 0.0             # tolerance on F_DPC increase before damping

    # --- Training (Algorithm 3) ----------------------------------------------
    epochs: int = 5
    eval_every: int = 1                      # evaluate every K epochs
    seed: int = 0

    # --- Evaluation (Section 6.6) --------------------------------------------
    mc_samples: int = 16                     # S in Eq. 103
    eval_T_z: Optional[int] = None           # target-free test-time E-step; None -> T_z
    eval_eta_m: Optional[float] = None       # None -> eta_m
    eval_eta_u: Optional[float] = None       # None -> eta_u
    eval_v_init: Optional[float] = None      # None -> v_init

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
        if self.objective not in ("pc_free_energy", "shared_dpc"):
            raise ValueError(f"objective must be 'pc_free_energy' or 'shared_dpc', got {self.objective!r}")
        if not 0.0 <= self.kappa_start <= 1.0:
            raise ValueError(f"kappa_start must be in [0, 1], got {self.kappa_start}")
        if self.kappa_warmup_epochs < 0:
            raise ValueError(f"kappa_warmup_epochs must be >= 0, got {self.kappa_warmup_epochs}")
        if self.rho_z < 0:
            raise ValueError(f"rho_z must be >= 0, got {self.rho_z}")
        if self.rho_w < 0:
            raise ValueError(f"rho_w must be >= 0, got {self.rho_w}")
        if self.r_max is not None and self.r_max <= 0:
            raise ValueError(f"r_max must be > 0 when set, got {self.r_max}")
        if not 0.0 < self.accept_damp_omega <= 1.0:
            raise ValueError(f"accept_damp_omega must be in (0, 1], got {self.accept_damp_omega}")

    @property
    def output_dim(self) -> int:
        return len(self.classes)

    @property
    def layer_dims(self) -> Tuple[int, ...]:
        return (self.input_dim, self.hidden_dim, self.output_dim)

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

    def kappa_for_epoch(self, epoch: int) -> float:
        """Compute kappa for a given epoch (1-indexed) via linear ramp.

        Reference: extension Eq. 63. kappa interpolates linearly from
        `kappa_start` at epoch 1 to 1.0 at epoch (kappa_warmup_epochs + 1),
        then stays at 1.0. When `kappa_warmup_epochs == 0`, kappa is 1.0
        from epoch 1 onward (no homotopy).
        """
        if self.kappa_warmup_epochs <= 0:
            return 1.0
        if epoch >= self.kappa_warmup_epochs + 1:
            return 1.0
        frac = (epoch - 1) / float(self.kappa_warmup_epochs)
        return float(self.kappa_start + frac * (1.0 - self.kappa_start))
