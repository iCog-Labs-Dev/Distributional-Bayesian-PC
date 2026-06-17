"""Numerical safeguards used throughout BPCN.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 8.1 (Eq. 111): log-variance parameterization sigma^2 = exp(tau).
- Section 7.5 "Variance collapse / explosion": bounds on tau.
- Eqs. 75-76: contain 1/v_p^2 -> v_p must be floored.
"""
import jax.numpy as jnp


EPS_V = 1e-8      # floor for predictive variance v_p (Eqs. 75-76 contain 1/v_p^2)
TAU_MIN = -12.0   # exp(-12) ~ 6e-6  (lower bound on sigma^2)
TAU_MAX = 6.0     # exp(6)   ~ 400   (upper bound on sigma^2)
U_MIN = -12.0     # lower bound on latent log-variance u (Eq. 33)
U_MAX = 6.0       # upper bound on latent log-variance u


def clamp_tau(tau):
    """Clamp log-variance of weight posterior (Section 7.5 mitigation)."""
    return jnp.clip(tau, TAU_MIN, TAU_MAX)


def clamp_u(u):
    """Clamp log-variance of latent posterior (Algorithm 1 step 4e)."""
    return jnp.clip(u, U_MIN, U_MAX)


def floor_v(v):
    """Floor predictive variance away from zero before any division (Eqs. 75-76)."""
    return jnp.maximum(v, EPS_V)
