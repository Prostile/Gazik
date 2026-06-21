from __future__ import annotations

from typing import Any

import numpy as np

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import GroundTruth


def validate_educational_objective(
    truth: GroundTruth,
    scenario: ScenarioConfig,
) -> dict[str, Any]:
    step_m = scenario.depth.step * (1.0 if scenario.depth.unit == "m" else 0.3048)
    pay_thickness = float(np.sum(truth.is_pay) * step_m)
    low, high = scenario.target.net_pay_thickness_m
    target_present = bool(
        scenario.target.hydrocarbon == "water" or (low - step_m <= pay_thickness <= high + step_m)
    )
    return {
        "valid": target_present,
        "target_learning_objective_present": target_present,
        "pay_thickness_m": pay_thickness,
        "configured_pay_range_m": [low, high],
        "hidden_truth_not_exposed_in_curves": True,
    }
