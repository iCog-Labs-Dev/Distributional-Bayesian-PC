""" Predictive-coding latent E-step (Algorithm 1) descending the shared energy. """

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp

from bpcn.models.network import Network
from bpcn.models.moments import moment_forward
from bpcn.inference.shared_energy import shared_free_energy
from bpcn.inference.feature_moments import psi_moments
from bpcn.utils.safe_math import clamp_u

 
class FrozenLatents(NamedTuple):
    """ Output of the E-step: stop-gradient-ed (mean, variance) per hidden layer. """
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
    m_initial: object = None     # Tuple[jax.Array, ...] length L_hidden
    v_initial: object = None     # Tuple[jax.Array, ...] length L_hidden


def initial_latents(net: Network, x, v_init: float, *,
                    perturb_std: float = 0.0, perturb_key=None):
    """ Initialize q(z^1), ..., q(z^L) by predictive feedforward. """
    
    L_hidden = net.L_hidden
    v_floor = jnp.asarray(v_init, dtype=x.dtype)
    m_h, v_h = x, jnp.zeros_like(x)
    m_list = []
    u_list = []
    if perturb_std > 0.0 and perturb_key is None:
        raise ValueError(
            "initial_latents: `perturb_key` is required when `perturb_std > 0`."
        )
    for l in range(L_hidden):
        m_pred, v_pred = moment_forward(net.layers[l], m_h, v_h)
        if perturb_std > 0.0:
            key_l = jax.random.fold_in(perturb_key, l)
            sigma_l = jnp.sqrt(jnp.maximum(v_pred, v_floor))
            noise = jax.random.normal(key_l, m_pred.shape, dtype=m_pred.dtype)
            m_pred = m_pred + jnp.asarray(perturb_std, dtype=m_pred.dtype) * sigma_l * noise
        u_pred = jnp.log(jnp.maximum(v_pred, v_floor))
        m_list.append(m_pred)
        u_list.append(u_pred)
        # Use the current layer's initialized mean as the next source; with
        # perturb_std > 0 this propagates the seed through the stack.
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
    y_idx=None,
    key=None,
    mc_samples_train: int = 1,
    # Optional predictive-disequilibrium seed.
    init_perturb_std: float = 0.0,
    init_perturb_key=None,
) -> Tuple[FrozenLatents, EStepDiagnostics]:
    """
    Run T_z latent gradient steps then freeze.

    Descends the shared free energy F_DPC with the weight
    pytree under `jax.lax.stop_gradient`. Assumption I4: no weight gradient
    flows out of this function.

    """
    W = jax.lax.stop_gradient(net)
    # Keep perturbation and E-step MC randomness independent.
    if init_perturb_std > 0.0 and init_perturb_key is None:
        if key is None:
            init_perturb_key = jax.random.PRNGKey(0)
        else:
            init_perturb_key, key = jax.random.split(key)
    m0, u0 = initial_latents(
        W, x, v_init,
        perturb_std=init_perturb_std,
        perturb_key=init_perturb_key,
    )

    if y_var is None:
        y_var = jnp.zeros_like(y)
    # Categorical-mode label safety: the y_idx=None fallback below substitutes
    # a dummy "class 0 for every example" tensor.
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

    m_zs = tuple(jax.lax.stop_gradient(m) for m in m_final)
    v_zs = tuple(jax.lax.stop_gradient(jnp.exp(u)) for u in u_final)

    m_initial_sg = tuple(jax.lax.stop_gradient(m) for m in m0)
    v_initial_sg = tuple(jax.lax.stop_gradient(jnp.exp(u)) for u in u0)

    diagnostics = EStepDiagnostics(
        F_trace=F_trace,
        F_initial=F_initial,
        F_final=F_trace[-1],
        m_initial=m_initial_sg,
        v_initial=v_initial_sg,
    )
    return FrozenLatents(m_zs=m_zs, v_zs=v_zs), diagnostics
