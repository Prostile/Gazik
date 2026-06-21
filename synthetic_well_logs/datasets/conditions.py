from __future__ import annotations

import numpy as np

CONDITION_LABELS = {
    "low_gr_clean_sand_like",
    "high_gr_shale_like",
    "mixed_shaly_sand_like",
    "dense_low_porosity_like",
    "resistive_pay_like",
    "unknown_mixed",
}


def window_curve_medians(
    curves: np.ndarray,
    curve_names: list[str],
    valid_mask: np.ndarray,
) -> dict[str, float]:
    medians: dict[str, float] = {}
    for channel, curve in enumerate(curve_names):
        values = curves[channel][valid_mask[channel] & np.isfinite(curves[channel])]
        medians[curve] = float(np.median(values)) if values.size else float("nan")
    return medians


def classify_electrofacies(
    medians: dict[str, float],
    *,
    rt_reference_p75: float | None = None,
) -> str:
    gr = medians.get("GR", float("nan"))
    rhob = medians.get("RHOB", float("nan"))
    nphi = medians.get("NPHI", float("nan"))
    rt = medians.get("RT", float("nan"))
    if not np.isfinite(gr):
        return "unknown_mixed"
    resistive_threshold = rt_reference_p75 if rt_reference_p75 is not None else 30.0
    if np.isfinite(rt) and rt > resistive_threshold and gr < 90:
        return "resistive_pay_like"
    if gr >= 90:
        return "high_gr_shale_like"
    if gr < 65 and 0.05 <= nphi <= 0.35 and 1.8 <= rhob <= 2.65:
        return "low_gr_clean_sand_like"
    if np.isfinite(rhob) and np.isfinite(nphi) and rhob >= 2.55 and nphi < 0.16:
        return "dense_low_porosity_like"
    if 55 <= gr <= 110:
        return "mixed_shaly_sand_like"
    return "unknown_mixed"
