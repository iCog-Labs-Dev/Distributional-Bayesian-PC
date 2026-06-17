"""KL between diagonal-Gaussian weight posterior and Gaussian prior."""
import jax.numpy as jnp


def gaussian_weight_kl(mu, tau, alpha, mu_0=0.0):
    """KL(N(mu, exp(tau)) || N(mu_0, alpha^2)) elementwise."""
    sigma2 = jnp.exp(tau)
    alpha2 = alpha * alpha
    return 0.5 * ((sigma2 + (mu - mu_0) ** 2) / alpha2
                  - 1.0
                  + jnp.log(alpha2) - tau)


def gaussian_weight_kl_components(mu, tau, alpha, mu_0=0.0):
    """Decomposed weight KL: (mu_term, var_term) such that their sum equals
    `gaussian_weight_kl(mu, tau, alpha, mu_0)`.

    The mu_term is non-negative; the var_term is non-negative with minimum
    zero at sigma^2 = alpha^2. Under a matched layerwise prior with
    sigma_w,0^2 = alpha_l^2 (per the continuation note's recommended fix),
    var_term is exactly zero at initialization, and the mu_term is the only
    residual contribution to the F_weight_kl floor.
    """
    sigma2 = jnp.exp(tau)
    alpha2 = alpha * alpha
    mu_term = 0.5 * (mu - mu_0) ** 2 / alpha2
    var_term = 0.5 * (sigma2 / alpha2 - 1.0 + jnp.log(alpha2) - tau)
    return mu_term, var_term
