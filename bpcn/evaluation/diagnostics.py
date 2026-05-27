"""Per-epoch diagnostics aggregator and variance decomposition logging.

References (write-up: distributional_predictive_coding_v2.pdf):
- Section 6.7 1-5.
- Eq. 87: v_pred decomposition into residual + propagated + epistemic.

References (extension: shared_energy_dbpcn_extension.pdf):
- Eq. 72: F_DPC = F_out + F_trans-DPC + F_weight-KL decomposition.
- Eq. 63: kappa-homotopy schedule logging.
"""
from typing import List, Dict, Optional
import numpy as np

from ..models.moments import variance_components


class EpochDiagnostics:
    """Accumulate per-minibatch scalars into per-epoch summaries."""

    def __init__(self):
        self.records: List[Dict] = []

    def add(self, e_diag, m_diag, f_dpc=None, kappa: Optional[float] = None):
        """Append one minibatch's diagnostics to the per-epoch record.

        Parameters
        ----------
        e_diag : EStepDiagnostics
            E-step trace and initial/final objective value.
        m_diag : dict {"hidden": LayerDiagnostics, "output": LayerDiagnostics}
        f_dpc : SharedEnergyTerms or None
            Decomposed F_DPC components after the M-step (extension Eq. 72).
            When the loop returns these (always in current implementation),
            they are recorded as F_out, F_trans_dpc, F_weight_kl. When None
            (legacy single-stage runs), the entries are omitted.
        kappa : float or None
            kappa-homotopy value used for this batch (extension Eq. 63).
            When None, omitted.
        """
        rec = {
            "F_initial": float(np.asarray(e_diag.F_initial)),
            "F_final":   float(np.asarray(e_diag.F_final)),
            "F_drop":    float(np.asarray(e_diag.F_initial - e_diag.F_final)),
            "F_monotone": float(np.all(np.diff(np.asarray(e_diag.F_trace)) <= 1e-3)),
        }
        if f_dpc is not None:
            rec["F_out"] = float(np.asarray(f_dpc.f_out))
            rec["F_trans_dpc"] = float(np.asarray(f_dpc.f_trans_dpc))
            rec["F_weight_kl"] = float(np.asarray(f_dpc.f_weight_kl))
            # Total F_DPC including the prior KL (extension Eq. 12 at kappa=1).
            rec["F_DPC_total"] = rec["F_out"] + rec["F_trans_dpc"] + rec["F_weight_kl"]
        if kappa is not None:
            rec["kappa"] = float(kappa)
        for name, ld in m_diag.items():
            rec[f"{name}/kl_data"] = float(np.asarray(ld.kl_data))
            rec[f"{name}/kl_data_before"] = float(np.asarray(ld.kl_data_before))
            rec[f"{name}/kl_data_after"] = float(np.asarray(ld.kl_data_after))
            rec[f"{name}/kl_data_delta"] = float(np.asarray(ld.kl_data_delta))
            rec[f"{name}/kl_data_loop_before"] = float(np.asarray(ld.kl_data_loop_before))
            rec[f"{name}/kl_data_loop_after"] = float(np.asarray(ld.kl_data_loop_after))
            rec[f"{name}/kl_data_loop_delta"] = float(np.asarray(ld.kl_data_loop_delta))
            rec[f"{name}/kl_weight"] = float(np.asarray(ld.kl_weight))
            rec[f"{name}/grad_mu_norm"] = float(np.asarray(ld.grad_mu_norm))
            rec[f"{name}/grad_tau_norm"] = float(np.asarray(ld.grad_tau_norm))
            rec[f"{name}/mean_sigma2"] = float(np.asarray(ld.mean_sigma2))
            rec[f"{name}/r_pos_frac"] = float(np.asarray(ld.var_residual_pos_frac))
            for k, v in ld.components.items():
                rec[f"{name}/components/{k}"] = float(np.asarray(v))
        self.records.append(rec)

    def summary(self) -> Dict[str, float]:
        if not self.records:
            return {}
        keys = self.records[0].keys()
        return {k: float(np.mean([r[k] for r in self.records])) for k in keys}


def variance_decomposition(net, m_h, v_h, layer_idx: int = -1):
    """Decompose v_pred at a given layer into the three components of Eq. 87.

    Used to verify V5: no single component absorbs all uncertainty.

    Note: `m_h`, `v_h` are the *presynaptic* feature moments for the chosen
    layer (post-psi). For the output layer with `psi != "identity"`, the
    caller must apply `psi_moments(net.psi, m_z, v_z)` before invoking this
    function; passing raw latent moments would mis-report the variance
    decomposition.
    """
    layer = net.layers[layer_idx]
    residual, propagated, epistemic = variance_components(layer, m_h, v_h)
    total = residual[None, :] + propagated + epistemic
    return {
        "residual_total": float(residual.sum()),
        "propagated_total": float(propagated.mean(axis=0).sum()),
        "epistemic_total": float(epistemic.mean(axis=0).sum()),
        "total_total": float(total.mean(axis=0).sum()),
        "residual_frac": float(residual.sum() / total.mean(axis=0).sum()),
        "propagated_frac": float(propagated.mean(axis=0).sum() / total.mean(axis=0).sum()),
        "epistemic_frac": float(epistemic.mean(axis=0).sum() / total.mean(axis=0).sum()),
    }
