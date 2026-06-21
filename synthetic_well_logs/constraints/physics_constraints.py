from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.constraints.consistency import expected_curves
from synthetic_well_logs.constraints.reports import FACIES_CURVE_RANGES
from synthetic_well_logs.domain import GroundTruth

RANGES = {
    "GR": (0.0, 250.0),
    "CALI": (5.0, 24.0),
    "RHOB": (1.5, 3.1),
    "NPHI": (-0.15, 0.8),
    "DT": (40.0, 200.0),
    "RT": (0.1, 10_000.0),
}

CONSISTENCY_TOLERANCE = {"GR": 40.0, "RHOB": 0.22, "NPHI": 0.20, "DT": 32.0}
CORRECTION_TOLERANCE = {"GR": 38.0, "RHOB": 0.20, "NPHI": 0.18, "DT": 30.0}


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
            values = result[curve].to_numpy(dtype=float)
            before[curve] = int(np.sum((values < lower) | (values > upper) | ~np.isfinite(values)))
            result[curve] = np.clip(values, lower, upper)

        expected = expected_curves(truth)
        for curve, tolerance in CORRECTION_TOLERANCE.items():
            if curve in result:
                result[curve] = np.clip(
                    result[curve].to_numpy(dtype=float),
                    expected[curve] - tolerance,
                    expected[curve] + tolerance,
                )
        if "NPHI" in result:
            gas = truth.fluid == "gas"
            nphi = result["NPHI"].to_numpy(copy=True)
            nphi[gas] = np.minimum(nphi[gas], truth.phi[gas] - 0.025)
            result["NPHI"] = np.clip(nphi, *RANGES["NPHI"])
        if "RT" in result:
            rt = result["RT"].to_numpy(copy=True)
            lower = np.maximum(expected["RT"] / 25.0, RANGES["RT"][0])
            upper = np.minimum(expected["RT"] * 25.0, RANGES["RT"][1])
            rt = np.clip(rt, lower, upper)
            rt[truth.is_pay] = np.maximum(rt[truth.is_pay], 2.0)
            result["RT"] = rt

        after = {
            curve: int(
                np.sum(
                    (result[curve].to_numpy(dtype=float) < bounds[0])
                    | (result[curve].to_numpy(dtype=float) > bounds[1])
                    | ~np.isfinite(result[curve].to_numpy(dtype=float))
                )
            )
            for curve, bounds in RANGES.items()
            if curve in result
        }
        consistency_rates = self._consistency_rates(result, truth, expected)
        facies_rates = self._facies_rates(result, truth)
        gas = (truth.fluid == "gas") & truth.is_pay
        gas_preserved = bool(
            not gas.any()
            or "NPHI" not in result
            or np.all(result["NPHI"].to_numpy(dtype=float)[gas] < truth.phi[gas])
        )
        pay_preserved = bool(
            not truth.is_pay.any()
            or "RT" not in result
            or np.all(result["RT"].to_numpy(dtype=float)[truth.is_pay] >= 2.0)
        )
        rate_values = list(consistency_rates.values())
        range_rate = sum(after.values()) / max(len(result) * max(len(after), 1), 1)
        total_rate = (
            float(np.mean([range_rate, *rate_values])) if rate_values else float(range_rate)
        )
        report: dict[str, object] = {
            "range_violations_before": before,
            "range_violations_after": after,
            "facies_range_violations": facies_rates,
            **consistency_rates,
            "gas_effect_preserved": gas_preserved,
            "pay_interval_preserved": pay_preserved,
            "constraint_violation_rate": total_rate,
        }
        return result, report

    @staticmethod
    def _consistency_rates(
        curves: pd.DataFrame,
        truth: GroundTruth,
        expected: dict[str, np.ndarray],
    ) -> dict[str, float]:
        mapping = {
            "GR": "gr_vsh_violation_rate",
            "RHOB": "rhob_phi_violation_rate",
            "NPHI": "nphi_phi_violation_rate",
            "DT": "dt_phi_vsh_violation_rate",
        }
        rates: dict[str, float] = {}
        for curve, report_name in mapping.items():
            if curve in curves:
                rates[report_name] = float(
                    np.mean(
                        np.abs(curves[curve].to_numpy(dtype=float) - expected[curve])
                        > CONSISTENCY_TOLERANCE[curve]
                    )
                )
        if "RT" in curves:
            ratio = curves["RT"].to_numpy(dtype=float) / np.maximum(expected["RT"], 1e-9)
            rates["rt_sw_violation_rate"] = float(np.mean((ratio < 1 / 30) | (ratio > 30)))
        return rates

    @staticmethod
    def _facies_rates(curves: pd.DataFrame, truth: GroundTruth) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for facies, curve_ranges in FACIES_CURVE_RANGES.items():
            mask = truth.facies == facies
            if not mask.any():
                continue
            output[facies] = {}
            for curve, (low, high) in curve_ranges.items():
                if curve in curves:
                    values = curves[curve].to_numpy(dtype=float)[mask]
                    output[facies][curve] = float(np.mean((values < low) | (values > high)))
        return output
