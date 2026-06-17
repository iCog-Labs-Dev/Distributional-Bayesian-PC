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
