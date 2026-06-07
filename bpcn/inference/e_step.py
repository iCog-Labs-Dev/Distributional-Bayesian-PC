"""Predictive-coding latent E-step (Algorithm 1) descending the shared energy.

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
- Eq. 12: F_DPC -- the shared energy this E-step descends.
- Eq. 22-23: E-step descends F_DPC, not the legacy F_z.
- Eq. 31: full latent update with target and source roles.
- Algorithm 1 (Section 6.1 of the extension): shared-energy latent E-step.

Assumption I4 (plan): autodiff over (m, u) buffers with the weight pytree
under stop_gradient. The E-step never updates weights.
"""
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from ..models.network import Network
from ..models.moments import moment_forward
from .shared_energy import shared_free_energy
from .feature_moments import psi_moments
from ..utils.safe_math import clamp_u


class FrozenLatents(NamedTuple):
    """Output of the E-step: stop-gradient-ed (mean, variance) per hidden layer.

    For multi-layer support the latents are carried as tuples of length
    `L_hidden`. `m_zs[l]` and `v_zs[l]` are the frozen mean and variance of
    hidden latent `z^{l+1}` (zero-indexed in code; matches v2 Eq. 7's
    `z^1, ..., z^L`).

    Backward-compat `.m_z` / `.v_z` properties return the top latent
    (`m_zs[-1]` / `v_zs[-1]`) so every downstream consumer that needs only
    the top hidden layer's posterior (predict path, output-head M-step)
    keeps working without code change.
    """
    m_zs: Tuple[jax.Array, ...]    # length L_hidden, each [B, d_l]
    v_zs: Tuple[jax.Array, ...]    # length L_hidden, each [B, d_l]

    @property
    def m_z(self) -> jax.Array:
        return self.m_zs[-1]

    @property
    def v_z(self) -> jax.Array:
        return self.v_zs[-1]


class EStepDiagnostics(NamedTuple):
    F_trace: jax.Array          # [T_z]   per-iteration objective value (mean over batch)
    F_initial: jax.Array        # scalar  objective at t=0
    F_final: jax.Array          # scalar  objective after T_z steps


def initial_latents(net: Network, x, v_init: float):
    """Initialize q(z^1), ..., q(z^L) by predictive feedforward (v2 Eq. 89).

    For each layer l in 0..L_hidden-1 (code-indexed):
      - Compute presynaptic feature moments (m_h, v_h):
          l == 0 : (x, zeros_like(x))    # input is deterministic, v2 Eq. 94
          l >= 1 : psi_moments(net.activations[l-1], m_prev, v_prev)
                                          # v2 Eq. 21 / Section 4.5
      - Compute predictive moments (m_pred, v_pred) = moment_forward(net.layers[l], m_h, v_h)
                                          # v2 Eqs. 60-61
      - Set m_l = m_pred (Eq. 89: feedforward mean) and
        u_l = log(max(v_pred, v_floor)).
    Returns tuples of length L_hidden.
    """
    L_hidden = net.L_hidden
    v_floor = jnp.asarray(v_init, dtype=x.dtype)
    m_h, v_h = x, jnp.zeros_like(x)
    m_list = []
    u_list = []
    for l in range(L_hidden):
        m_pred, v_pred = moment_forward(net.layers[l], m_h, v_h)
        u_pred = jnp.log(jnp.maximum(v_pred, v_floor))
        m_list.append(m_pred)
        u_list.append(u_pred)
        # Prepare next layer's presynaptic feature moments via psi_l.
        if l + 1 < L_hidden:
            m_h, v_h = psi_moments(net.activations[l], m_pred, v_pred)
    return tuple(m_list), tuple(u_list)


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
    y_var=None,
    gamma_hidden: float = 1.0,
    gamma_output: float = 1.0,
    # Categorical-output kwargs (continuation note). Consumed only when
    # `net.output_likelihood == "categorical"`; harmless under "gaussian".
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
) -> Tuple[FrozenLatents, EStepDiagnostics]:
    """Run T_z latent gradient steps then freeze (Algorithm 1 / extension Alg. 1).

    Descends the shared free energy F_DPC (extension Eq. 12) with the weight
    pytree under `jax.lax.stop_gradient`. Assumption I4: no weight gradient
    flows out of this function.

    Parameters
    ----------
    y : [B, C]
        Output target as a mean vector. For Gaussian-logit targets (I3),
        this is `y_mean` and the caller should also pass `y_var`.
    output_weight : float
        Multiplier on the output target term (K_out, extension Eq. 12 first
        sum). 1.0 (default) is Algorithm 1; 0.0 is target-free test-time
        inference (Section 6.6 paragraph 1). Under the categorical head this
        also plays the role of `lambda_y` (continuation Eq. 2/48).
    y_var : [B, C] or None
        Output target variance for Gaussian-logit targets (assumption I3).
        If None, defaults to a zero array (deterministic targets).
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

    def F_of_log_var(m, u, k):
        # m, u are tuples of length L_hidden; jax.tree.map applies exp per layer.
        v = jax.tree.map(jnp.exp, u)
        return shared_free_energy(
            W, x, y, y_var, m, v,
            y_idx=y_idx, key=k, mc_samples_train=mc_samples_train,
            gamma_hidden=gamma_hidden,
            gamma_output=gamma_output,
            include_weight_kl=False,
            output_weight=output_weight,
        )

    grad_fn = jax.grad(F_of_log_var, argnums=(0, 1))

    def step(carry, k):
        m, u = carry
        gm, gu = grad_fn(m, u, k)
        # Per-layer descent step on the tuple of latents (extension Eq. 57).
        m_new = jax.tree.map(lambda mi, gmi: mi - eta_m * gmi, m, gm)
        u_new = jax.tree.map(lambda ui, gui: clamp_u(ui - eta_u * gui), u, gu)
        F_new = F_of_log_var(m_new, u_new, k)
        return (m_new, u_new), F_new

    # Fresh key per inner-loop iteration so MC samples are independent
    # across the T_z steps. Under Gaussian / MEAN the keys are unused but
    # threading them keeps the compiled graph constant.
    keys_scan = jax.random.split(key, T_z)
    (m_final, u_final), F_trace = jax.lax.scan(step, (m0, u0), keys_scan, length=T_z)
    F_initial = F_of_log_var(m0, u0, keys_scan[0])

    # Freeze (v2 Eq. 92 / continuation Eq. 39): stop_gradient on every layer's
    # (mean, variance) so the M-step treats them as fixed numerical targets.
    m_zs = tuple(jax.lax.stop_gradient(m) for m in m_final)
    v_zs = tuple(jax.lax.stop_gradient(jnp.exp(u)) for u in u_final)

    diagnostics = EStepDiagnostics(
        F_trace=F_trace,
        F_initial=F_initial,
        F_final=F_trace[-1],
    )
    return FrozenLatents(m_zs=m_zs, v_zs=v_zs), diagnostics
