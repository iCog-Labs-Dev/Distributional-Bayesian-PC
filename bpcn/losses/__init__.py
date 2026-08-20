"""Loss functions used by BPCN."""

from bpcn.losses.categorical import categorical_output_loss
from bpcn.losses.distributional_kl import GaussianKLTerms, gaussian_kl
from bpcn.losses.weight_kl import (
    WeightKLComponents,
    gaussian_weight_kl,
    gaussian_weight_kl_components,
)

__all__ = [
    "GaussianKLTerms",
    "WeightKLComponents",
    "categorical_output_loss",
    "gaussian_kl",
    "gaussian_weight_kl",
    "gaussian_weight_kl_components",
]
