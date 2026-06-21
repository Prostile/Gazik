from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.datasets.calibration_dataset import CalibrationDataset
from synthetic_well_logs.datasets.reports import calibration_frame, curve_statistics
from synthetic_well_logs.validation.statistical_gate import evaluate_statistical_gate


def statistical_validation_status(
    realism_report: dict[str, Any] | None,
    curves: pd.DataFrame,
    scenario: ScenarioConfig,
) -> dict[str, Any]:
    report: dict[str, Any] = dict(realism_report or {})
    if not scenario.realism.statistical_gate:
        gate = {"status": "not_checked", "valid": True, "reason": "statistical gate is disabled"}
    elif not scenario.realism.calibration_dataset_path:
        gate = {
            "status": "not_checked",
            "valid": True,
            "reason": "calibration dataset path is not configured",
        }
    elif not Path(scenario.realism.calibration_dataset_path).exists():
        gate = {
            "status": "not_checked",
            "valid": True,
            "reason": "calibration dataset path does not exist",
        }
    else:
        dataset = CalibrationDataset.open(scenario.realism.calibration_dataset_path)
        real_stats = curve_statistics(calibration_frame(dataset), dataset.curve_names)
        synthetic_stats = curve_statistics(curves, dataset.curve_names)
        gate = evaluate_statistical_gate(real_stats, synthetic_stats)
    report["statistical_gate"] = gate
    report.setdefault("status", "model_applied" if realism_report else "not_calibrated")
    report["valid"] = bool(gate["valid"])
    return report
