"""One-hot Gaussian logit target encoding (assumption I3).

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 8.2 Option 2: "introduce Gaussian logit targets and match distributions
  in logit space".
- Section 3.1 paragraph after Eq. 26: regression / Gaussian pseudo-output
  preferred for initial implementations.

The target for the output layer M-step is a frozen Gaussian posterior with
  m_z^{out}_n = onehot(y_n),     v_z^{out}_n = epsilon_y .
"""
from typing import Tuple
import numpy as np


def gaussian_logit_target(
    y_idx: np.ndarray,
    n_classes: int,
    epsilon_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode class indices as Gaussian one-hot targets.

    Parameters
    ----------
    y_idx : [B], integer in [0, n_classes).
    n_classes : C.
    epsilon_y : variance assigned to the target (assumption I3).

    Returns
    -------
    y_mean : [B, C]  one-hot.
    y_var  : [B, C]  filled with epsilon_y (constant).
    """
    B = y_idx.shape[0]
    y_mean = np.zeros((B, n_classes), dtype=np.float32)
    y_mean[np.arange(B), y_idx] = 1.0
    y_var = np.full((B, n_classes), float(epsilon_y), dtype=np.float32)
    return y_mean, y_var
