from __future__ import annotations

import numpy as np

GLOBAL_RANGES = {
    "GR": (0.0, 250.0),
    "RHOB": (1.5, 3.1),
    "NPHI": (-0.15, 0.8),
    "DT": (40.0, 200.0),
    "RT": (0.1, 10_000.0),
}


def candidate_constraint_score(
    curves: dict[str, np.ndarray],
    truth_slice: dict[str, np.ndarray],
) -> float:
    rates: list[float] = []
    for curve, values in curves.items():
        if curve not in GLOBAL_RANGES:
            continue
        low, high = GLOBAL_RANGES[curve]
        rates.append(float(np.mean((values < low) | (values > high) | ~np.isfinite(values))))

    if "GR" in curves:
        expected = 18.0 + 132.0 * truth_slice["vsh"]
        rates.append(float(np.mean(np.abs(curves["GR"] - expected) > 45.0)))
    if "RHOB" in curves:
        rates.append(float(np.mean(np.abs(np.diff(curves["RHOB"])) > 0.45)))
    if "RT" in curves:
        pay = truth_slice["is_pay"]
        if pay.any():
            rates.append(float(np.mean(curves["RT"][pay] < 2.0)))
    if "NPHI" in curves:
        gas = truth_slice["fluid"] == "gas"
        if gas.any():
            rates.append(float(np.mean(curves["NPHI"][gas] >= truth_slice["phi"][gas])))
    return float(np.mean(rates)) if rates else 0.0
