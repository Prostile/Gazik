from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from synthetic_well_logs.config import ResistivityModelConfig
from synthetic_well_logs.constraints.reports import FACIES_CURVE_RANGES
from synthetic_well_logs.domain import GroundTruth
from synthetic_well_logs.forward.model import calculate_expected_curves_from_arrays

GLOBAL_RANGES = {
    "GR": (0.0, 250.0),
    "CALI": (5.0, 24.0),
    "RHOB": (1.1, 3.1),
    "NPHI": (-0.15, 0.8),
    "DT": (40.0, 200.0),
    "RT": (0.1, 10_000.0),
}

DEFAULT_SCORE_WEIGHTS = {
    "global_range_violation": 1.0,
    "gr_vsh_violation": 1.0,
    "rhob_phi_violation": 1.0,
    "nphi_phi_violation": 1.0,
    "dt_phi_vsh_violation": 0.8,
    "rt_sw_violation": 1.2,
    "facies_range_violation": 0.8,
    "gas_effect_violation": 1.5,
    "pay_interval_violation": 2.0,
    "nonfinite_violation": 2.0,
}


@dataclass(frozen=True, slots=True)
class TruthSlice:
    vsh: np.ndarray
    phi: np.ndarray
    sw: np.ndarray
    facies: np.ndarray
    lithology: np.ndarray
    fluid: np.ndarray
    is_pay: np.ndarray
    is_reservoir: np.ndarray

    @classmethod
    def from_ground_truth(
        cls,
        truth: GroundTruth,
        start: int = 0,
        stop: int | None = None,
    ) -> TruthSlice:
        selection = slice(start, stop)
        return cls(
            vsh=truth.vsh[selection],
            phi=truth.phi[selection],
            sw=truth.sw[selection],
            facies=truth.facies[selection],
            lithology=truth.lithology[selection],
            fluid=truth.fluid[selection],
            is_pay=truth.is_pay[selection],
            is_reservoir=truth.is_reservoir[selection],
        )


@dataclass(frozen=True, slots=True)
class ConstraintScore:
    total: float
    components: dict[str, float]
    valid: bool


def expected_curves_for_slice(
    truth: TruthSlice,
    resistivity_config: ResistivityModelConfig | None = None,
) -> dict[str, np.ndarray]:
    return calculate_expected_curves_from_arrays(
        vsh=truth.vsh,
        phi=truth.phi,
        sw=truth.sw,
        lithology=truth.lithology,
        fluid=truth.fluid,
        resistivity_config=resistivity_config,
    )


def score_curves_against_truth(
    curves: Mapping[str, np.ndarray],
    truth_slice: TruthSlice,
    *,
    resistivity_config: ResistivityModelConfig | None = None,
    include_facies_ranges: bool = True,
    include_pay_checks: bool = True,
    include_gas_checks: bool = True,
) -> ConstraintScore:
    values = {name: np.asarray(value, dtype=float) for name, value in curves.items()}
    expected = expected_curves_for_slice(truth_slice, resistivity_config)
    components = {name: 0.0 for name in DEFAULT_SCORE_WEIGHTS}
    finite_rates: list[float] = []
    range_rates: list[float] = []
    for curve, array in values.items():
        finite = np.isfinite(array)
        finite_rates.append(float(np.mean(~finite)))
        if curve in GLOBAL_RANGES:
            low, high = GLOBAL_RANGES[curve]
            range_rates.append(float(np.mean(finite & ((array < low) | (array > high)))))
    components["nonfinite_violation"] = float(np.mean(finite_rates)) if finite_rates else 0.0
    components["global_range_violation"] = float(np.mean(range_rates)) if range_rates else 0.0
    if "GR" in values:
        components["gr_vsh_violation"] = float(np.mean(np.abs(values["GR"] - expected["GR"]) > 40))
    if "RHOB" in values:
        components["rhob_phi_violation"] = float(
            np.mean(np.abs(values["RHOB"] - expected["RHOB"]) > 0.22)
        )
    if "NPHI" in values:
        components["nphi_phi_violation"] = float(
            np.mean(np.abs(values["NPHI"] - expected["NPHI"]) > 0.20)
        )
    if "DT" in values:
        components["dt_phi_vsh_violation"] = float(
            np.mean(np.abs(values["DT"] - expected["DT"]) > 32)
        )
    if "RT" in values:
        ratio = values["RT"] / np.maximum(expected["RT"], 1e-9)
        components["rt_sw_violation"] = float(np.mean((ratio < 1 / 30) | (ratio > 30)))
    if include_facies_ranges:
        rates: list[float] = []
        for facies, curve_ranges in FACIES_CURVE_RANGES.items():
            mask = truth_slice.facies == facies
            if not mask.any():
                continue
            for curve, (low, high) in curve_ranges.items():
                if curve in values:
                    rates.append(
                        float(np.mean((values[curve][mask] < low) | (values[curve][mask] > high)))
                    )
        components["facies_range_violation"] = float(np.mean(rates)) if rates else 0.0
    if include_gas_checks and "NPHI" in values:
        gas = truth_slice.fluid == "gas"
        if gas.any():
            components["gas_effect_violation"] = float(
                np.mean(values["NPHI"][gas] >= truth_slice.phi[gas])
            )
    if include_pay_checks and truth_slice.is_pay.any() and "RT" in values:
        components["pay_interval_violation"] = float(
            np.mean(values["RT"][truth_slice.is_pay] < 2.0)
        )
    weighted = sum(components[name] * weight for name, weight in DEFAULT_SCORE_WEIGHTS.items())
    total_weight = sum(DEFAULT_SCORE_WEIGHTS.values())
    total = float(weighted / total_weight)
    return ConstraintScore(total=total, components=components, valid=total <= 0.05)
