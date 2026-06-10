"""KL between diagonal-Gaussian weight posterior and Gaussian prior.

References (write-up: distributional_predictive_coding_v2.pdf):
- Eq. 27: prior p(W) = prod N(0, alpha^2).
- Eq. 77: KL(q(w) || p(w)) for w ~ N(mu, sigma^2), p ~ N(0, alpha^2):
    = 0.5 [ (sigma^2 + mu^2)/alpha^2  - 1  + log(alpha^2 / sigma^2) ].

Generalized form (used for plan §4 option (b) posterior-as-prior) accepts
optional non-zero prior mean mu_0; the form is then
    0.5 [ (sigma^2 + (mu - mu_0)^2)/alpha^2  - 1  + log(alpha^2 / sigma^2) ].
"""
import jax.numpy as jnp


def gaussian_weight_kl(mu, tau, alpha, mu_0=0.0):
    """KL(N(mu, exp(tau)) || N(mu_0, alpha^2)) elementwise (Eq. 77)."""
    sigma2 = jnp.exp(tau)
    alpha2 = alpha * alpha
    return 0.5 * ((sigma2 + (mu - mu_0) ** 2) / alpha2
                  - 1.0
                  + jnp.log(alpha2) - tau)


def gaussian_weight_kl_components(mu, tau, alpha, mu_0=0.0):
    """Decomposed weight KL: (mu_term, var_term) such that their sum equals
    `gaussian_weight_kl(mu, tau, alpha, mu_0)`.

    References (DBPCN/dbpcn_weight_kl_sigma2_continuation.pdf):
    - §5 step 2: "Always report F^mu_weight-KL, F^sigma_weight-KL" — i.e.,
      decompose the total KL into a mean-displacement component and a
      variance/log-ratio component.

    Decomposition of v2 Eq. 77:
      KL = 0.5 * [(sigma^2 + (mu - mu_0)^2) / alpha^2  - 1 + log(alpha^2 / sigma^2)]
         = 0.5 * (mu - mu_0)^2 / alpha^2                                  <- mu_term
           + 0.5 * (sigma^2 / alpha^2 - 1 + log(alpha^2 / sigma^2))       <- var_term

    The mu_term is non-negative; the var_term is non-negative with minimum
    zero at sigma^2 = alpha^2. Under a matched layerwise prior with
    sigma_w,0^2 = alpha_l^2 (per the continuation note's recommended fix),
    var_term is exactly zero at initialization, and the mu_term is the only
    residual contribution to the F_weight_kl floor.

    Returns
    -------
    (mu_term, var_term) : tuple of jax.Array
        Same shape as `mu` / `tau`. Sum matches `gaussian_weight_kl` to
        machine precision.
    """
    sigma2 = jnp.exp(tau)
    alpha2 = alpha * alpha
    mu_term = 0.5 * (mu - mu_0) ** 2 / alpha2
    var_term = 0.5 * (sigma2 / alpha2 - 1.0 + jnp.log(alpha2) - tau)
    return mu_term, var_term
