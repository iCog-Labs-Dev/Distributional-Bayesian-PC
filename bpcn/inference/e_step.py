"""Predictive-coding latent E-step (Algorithm 1) with optional shared-energy dispatch.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 4.3 / Algorithm 1 (Section 6.3).
- Eq. 41: latent sampling z = m + s * xi; we use deterministic moment energies instead.
- Eq. 43: gradient steps on (m, u) where u = log s^2.
- Eq. 47-48: schematic local PC gradients.
- Eq. 49 / Eq. 92: freeze frozen latent statistics via stop_gradient.
- Eq. 89: initialise m by feedforward of mu.
- Predictive-init extension: initialise u from the moment-matched local
  Bayesian prior variance, floored by v_init.

References (extension: shared_energy_dbpcn_extension.pdf):
- Eq. 22-23: E-step descends F_DPC, not the legacy F_z.
- Eq. 31: full latent update with target and source roles.
- Eq. 63: kappa-homotopy F_kappa.
- Eq. 65: proximal latent damping F_E-prox.
- Algorithm 1 (Section 6.1 of the extension): shared-energy latent E-step.

Assumption I4 (plan): autodiff over (m, u) buffers with the weight pytree
under stop_gradient. The E-step never updates weights.

The `objective` kwarg selects between:
- "pc_free_energy" : legacy Eq. 40 free energy.
- "shared_dpc"     : extension Eq. 63 F_kappa shared energy.
"""
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from ..models.network import Network
from ..models.moments import moment_forward
from ..losses.distributional_kl import gaussian_kl
from .free_energy import free_energy
from .shared_energy import shared_free_energy
from ..utils.safe_math import clamp_u


class FrozenLatents(NamedTuple):
    """Output of the E-step: stop-gradient-ed (mean, variance) of the hidden latent."""
    m_z: jax.Array     # [B, d_1]
    v_z: jax.Array     # [B, d_1]


class EStepDiagnostics(NamedTuple):
    F_trace: jax.Array          # [T_z]   per-iteration objective value (mean over batch)
    F_initial: jax.Array        # scalar  objective at t=0
    F_final: jax.Array          # scalar  objective after T_z steps


def initial_latents(net: Network, x, v_init: float):
    """Initialize q(z^1) from the hidden layer predictive prior.

    m^1 is still the posterior-mean feedforward prediction (Eq. 89). The
    variance now uses the moment-matched predictive variance from Eqs. 60-61,
    with v_init retained as a numerical floor rather than the actual constant
    initialization.
    """
    hidden = net.layers[0]
    m, v_pred = moment_forward(hidden, x, jnp.zeros_like(x))  # [B, d_1]
    v_floor = jnp.asarray(v_init, dtype=v_pred.dtype)
    u = jnp.log(jnp.maximum(v_pred, v_floor))                 # log latent variance
    return m, u


def e_step(
    net: Network,
    x: jax.Array,
    y: jax.Array,
    *,
    T_z: int,
    eta_m: float,
    eta_u: float,
    v_init: float,
    output_weight: float = 1.0,
    # Shared-energy extension kwargs (defaults preserve legacy behaviour).
    y_var=None,
    objective: str = "pc_free_energy",
    kappa: float = 1.0,
    rho_z: float = 0.0,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
    # Categorical-output kwargs (continuation note). Consumed only when
    # `net.output_likelihood == "categorical"`; harmless under "gaussian".
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
) -> Tuple[FrozenLatents, EStepDiagnostics]:
    """Run T_z latent gradient steps then freeze (Algorithm 1 / extension Alg. 1).

    Weights enter under stop_gradient (assumption I4); no weight gradient flows
    out of this function.

    Parameters
    ----------
    y : [B, C]
        Output target as a mean vector. For Gaussian-logit targets (I3),
        this is `y_mean` and the caller should also pass `y_var`.
    output_weight : float
        Multiplier on the output target term. 1.0 (default) is Algorithm 1;
        0.0 is target-free test-time inference (Section 6.6 paragraph 1).
        Applies to both `objective="pc_free_energy"` (scales legacy output
        likelihood) and `objective="shared_dpc"` (scales the K_out shared-KL
        term, parallel to legacy behavior).
    y_var : [B, C] or None
        Output target variance for Gaussian-logit targets (assumption I3).
        Required when `objective == "shared_dpc"`. If None, defaults to a
        zero array (deterministic targets) which is mathematically equivalent
        to the legacy output term at output_weight=1.
    objective : str
        Which scalar to descend:
        - "pc_free_energy": legacy Eq. 40 free energy (NLL + entropy).
        - "shared_dpc"    : extension Eq. 63 F_kappa.
    kappa : float in [0, 1]
        kappa-homotopy mixing weight for the hidden transition (Eq. 63).
        Ignored when objective == "pc_free_energy".
    rho_z : float, >= 0
        Proximal latent damping coefficient (extension Eq. 65). 0 = off.
        Anchors the latent posterior to its feedforward initialization
        (m_init, u_init) for each scan iteration.
    gamma_hidden, gamma_output : float
        Per-layer prior-KL coefficients gamma_l. Forwarded to shared energy
        for include_weight_kl=False (E-step does not need the constant
        weight-KL term in its scalar; the M-step uses these via m_step).
    y_idx, key, mc_samples_train
        Categorical-output kwargs. Required when `net.output_likelihood ==
        "categorical"`. `y_idx` is the [B] integer-class target. `key` is
        the PRNG key for MC sampling; the inner scan splits one fresh key
        per iteration so each T_z step uses independent randomness. Under
        `output_likelihood == "gaussian"` these are ignored.
    """
    if objective not in ("pc_free_energy", "shared_dpc"):
        raise ValueError(f"unknown objective: {objective!r}")

    W = jax.lax.stop_gradient(net)
    m0, u0 = initial_latents(W, x, v_init)

    if y_var is None:
        y_var = jnp.zeros_like(y)
    # Categorical-mode label safety: the y_idx=None fallback below substitutes
    # a dummy "class 0 for every example" tensor. That is only safe when the
    # output term is disabled (output_weight == 0; v2 Section 6.6 target-free
    # inference) -- with a non-zero output term it silently grades the entire
    # batch against class 0 and produces wrong latent updates. Surface the
    # caller error explicitly instead of letting it slip past.
    if (
        net.output_likelihood == "categorical"
        and y_idx is None
        and float(output_weight) != 0.0
    ):
        raise ValueError(
            "e_step requires `y_idx` when net.output_likelihood == 'categorical' "
            "and output_weight != 0. Pass the integer class targets explicitly, "
            "or set output_weight=0.0 for target-free inference (Section 6.6)."
        )
    # Placeholder y_idx for Gaussian mode (the F call is shape-stable) and for
    # categorical target-free inference (the dummy is multiplied by 0).
    if y_idx is None:
        B = y.shape[0]
        y_idx = jnp.zeros((B,), dtype=jnp.int32)
    # Always feed a real key into the F call so the MC reparam branch can
    # split it. Gaussian and MEAN paths ignore the key.
    if key is None:
        key = jax.random.PRNGKey(0)

    use_prox = float(rho_z) != 0.0
    # Stop-gradient through prox anchors so they don't flow back into params.
    m_init = jax.lax.stop_gradient(m0) if use_prox else None
    u_init = jax.lax.stop_gradient(u0) if use_prox else None

    def F_of_log_var(m, u, k):
        v = jnp.exp(u)
        if objective == "shared_dpc":
            F = shared_free_energy(
                W, x, y, y_var, m, v,
                y_idx=y_idx, key=k, mc_samples_train=mc_samples_train,
                kappa=kappa,
                gamma_hidden=gamma_hidden,
                gamma_output=gamma_output,
                include_weight_kl=False,
                output_weight=output_weight,
            )
        else:  # "pc_free_energy"
            F = free_energy(
                W, x, y, m, v,
                output_weight=output_weight,
                y_idx=y_idx, key=k, mc_samples_train=mc_samples_train,
            )
        if use_prox:
            # Proximal latent damping (extension Eq. 65). Anchored at the
            # feedforward initial (m_init, u_init).
            v_init_anchor = jnp.exp(u_init)
            prox = gaussian_kl(m_z=m, v_z=v, m_p=m_init, v_p=v_init_anchor).kl.sum(axis=-1).mean()
            F = F + float(rho_z) * prox
        return F

    grad_fn = jax.grad(F_of_log_var, argnums=(0, 1))

    def step(carry, k):
        m, u = carry
        gm, gu = grad_fn(m, u, k)
        m_new = m - eta_m * gm                              # Eq. 43 / extension Eq. 57
        u_new = clamp_u(u - eta_u * gu)                     # Eq. 43 + Algorithm 1 step 4e
        F_new = F_of_log_var(m_new, u_new, k)
        return (m_new, u_new), F_new

    # Fresh key per inner-loop iteration so MC samples are independent
    # across the T_z steps. Under Gaussian / MEAN the keys are unused but
    # threading them keeps the compiled graph constant.
    keys_scan = jax.random.split(key, T_z)
    (m_final, u_final), F_trace = jax.lax.scan(step, (m0, u0), keys_scan, length=T_z)
    F_initial = F_of_log_var(m0, u0, keys_scan[0])

    # Freeze (Eq. 49 / Eq. 92): stop_gradient + record final statistics.
    m_z = jax.lax.stop_gradient(m_final)
    v_z = jax.lax.stop_gradient(jnp.exp(u_final))

    diagnostics = EStepDiagnostics(
        F_trace=F_trace,
        F_initial=F_initial,
        F_final=F_trace[-1],
    )
    return FrozenLatents(m_z=m_z, v_z=v_z), diagnostics
