"""Per-epoch diagnostics aggregator and variance decomposition logging."""

from typing import List, Dict
import numpy as np

from bpcn.models.moments import variance_components


class EpochDiagnostics:
    """Accumulate per-minibatch scalars into per-epoch summaries."""

    def __init__(self):
        self.records: List[Dict] = []

    def add(self, e_diag, m_diag, f_dpc=None,
            init_residuals=None, freeze_residuals=None,
            weight_kl_initial=None):
        """
        Append one minibatch's diagnostics to the per-epoch record.

        e_diag : EStepDiagnostics
            E-step trace and initial/final objective value.
        m_diag : dict {"hidden": LayerDiagnostics, "output": LayerDiagnostics}
        f_dpc : SharedEnergyTerms or None
            Decomposed F_DPC components after the M-step.
        init_residuals, freeze_residuals : dict[int, dict] or None
            Per-hidden-layer distributional residual diagnostics from
            `bpcn.inference.shared_energy.per_layer_residuals`. When provided,
        weight_kl_initial : float or None
            Initial (pre-training) value of F_weight_kl in per-data-point
            scale. When provided AND `f_dpc` is provided, emit the centred
            `F_weight_kl_delta = F_weight_kl_current - weight_kl_initial` key
            per the continuation note's recommended diagnostic
            (centred ΔKL is interpretable as Bayesian complexity added by
            training, where raw KL is dominated by the prior-vs-init scale
            mismatch). When None, the delta key is omitted.
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
            # Sum-identity: F_weight_kl_mu + F_weight_kl_var = F_weight_kl.
            rec["F_weight_kl_mu"] = float(np.asarray(f_dpc.f_weight_kl_mu))
            rec["F_weight_kl_var"] = float(np.asarray(f_dpc.f_weight_kl_var))
            # Total F_DPC = F_out + F_trans-DPC + F_weight-KL
            rec["F_DPC_total"] = rec["F_out"] + rec["F_trans_dpc"] + rec["F_weight_kl"]
            if weight_kl_initial is not None:
                rec["F_weight_kl_delta"] = rec["F_weight_kl"] - float(weight_kl_initial)
        if init_residuals is not None:
            for l, d in init_residuals.items():
                rec[f"hidden_{l}/init/K"] = float(np.asarray(d["K"]))
                rec[f"hidden_{l}/init/e_abs"] = float(np.asarray(d["e_abs"]))
                rec[f"hidden_{l}/init/r_abs"] = float(np.asarray(d["r_abs"]))
                rec[f"hidden_{l}/init/r_pos_frac"] = float(np.asarray(d["r_pos_frac"]))
                rec[f"hidden_{l}/init/r_pos_mag"] = float(np.asarray(d.get("r_pos_mag", 0.0)))
                rec[f"hidden_{l}/init/r_neg_mag"] = float(np.asarray(d.get("r_neg_mag", 0.0)))
        if freeze_residuals is not None:
            for l, d in freeze_residuals.items():
                rec[f"hidden_{l}/freeze/K"] = float(np.asarray(d["K"]))
                rec[f"hidden_{l}/freeze/e_abs"] = float(np.asarray(d["e_abs"]))
                rec[f"hidden_{l}/freeze/r_abs"] = float(np.asarray(d["r_abs"]))
                rec[f"hidden_{l}/freeze/r_pos_frac"] = float(np.asarray(d["r_pos_frac"]))
                rec[f"hidden_{l}/freeze/r_pos_mag"] = float(np.asarray(d.get("r_pos_mag", 0.0)))
                rec[f"hidden_{l}/freeze/r_neg_mag"] = float(np.asarray(d.get("r_neg_mag", 0.0)))
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
            gmd = float(np.asarray(ld.grad_mu_data_norm))
            gmp = float(np.asarray(ld.grad_mu_prior_norm))
            gtd = float(np.asarray(ld.grad_tau_data_norm))
            gtp = float(np.asarray(ld.grad_tau_prior_norm))
            rec[f"{name}/grad_mu_data_norm"] = gmd
            rec[f"{name}/grad_mu_prior_norm"] = gmp
            rec[f"{name}/grad_tau_data_norm"] = gtd
            rec[f"{name}/grad_tau_prior_norm"] = gtp
            rec[f"{name}/R_tau"] = gtd / max(gtp, 1e-12)
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
