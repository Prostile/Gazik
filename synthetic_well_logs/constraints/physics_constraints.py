from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import GroundTruth

RANGES = {
    "GR": (0.0, 250.0),
    "CALI": (5.0, 24.0),
    "RHOB": (1.5, 3.1),
    "NPHI": (-0.15, 0.8),
    "DT": (40.0, 200.0),
    "RT": (0.1, 10_000.0),
}


class PhysicsConstraints:
    def apply(
        self,
        curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        del scenario
        result = curves.copy(deep=True)
        before: dict[str, int] = {}
        for curve, (lower, upper) in RANGES.items():
            if curve not in result:
                continue
            values = result[curve].to_numpy()
            before[curve] = int(np.sum((values < lower) | (values > upper)))
            result[curve] = np.clip(values, lower, upper)

        if "GR" in result:
            expected = 18.0 + 132.0 * truth.vsh
            result["GR"] = np.clip(result["GR"], expected - 38.0, expected + 38.0)
        if "NPHI" in result:
            gas = truth.fluid == "gas"
            nphi = result["NPHI"].to_numpy(copy=True)
            nphi[gas] = np.minimum(nphi[gas], truth.phi[gas] - 0.025)
            result["NPHI"] = np.clip(nphi, *RANGES["NPHI"])
        if "RT" in result:
            hydrocarbon = (truth.fluid != "water") & truth.is_reservoir
            rt = result["RT"].to_numpy(copy=True)
            rt[hydrocarbon] = np.maximum(rt[hydrocarbon], 2.0)
            result["RT"] = np.clip(rt, *RANGES["RT"])

        after = {
            curve: int(
                np.sum((result[curve].to_numpy() < bounds[0]) | (result[curve] > bounds[1]))
            )
            for curve, bounds in RANGES.items()
            if curve in result
        }
        report = {
            "range_violations_before": before,
            "range_violations_after": after,
            "constraint_violation_rate": float(sum(after.values()) / max(result.size, 1)),
        }
        return result, report
