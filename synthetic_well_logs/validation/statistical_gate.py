from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class StatisticalGateConfig:
    max_mean_z_delta: float = 1.5
    max_std_ratio_delta: float = 1.0
    max_correlation_distance: float = 0.45
    max_autocorrelation_delta: float = 0.35
    min_passed_curves_fraction: float = 0.7


def evaluate_statistical_gate(
    real_stats: dict[str, Any],
    synthetic_stats: dict[str, Any],
    config: StatisticalGateConfig | None = None,
) -> dict[str, Any]:
    config = config or StatisticalGateConfig()
    curve_results: dict[str, dict[str, float | bool]] = {}
    common = sorted(set(real_stats["curves"]) & set(synthetic_stats["curves"]))
    for curve in common:
        real = real_stats["curves"][curve]
        synthetic = synthetic_stats["curves"][curve]
        if not real.get("count") or not synthetic.get("count") or real.get("std", 0) <= 1e-12:
            curve_results[curve] = {"passed": False, "insufficient_data": True}
            continue
        real_std = float(real["std"])
        synthetic_std = max(float(synthetic["std"]), 1e-12)
        mean_z = abs(float(synthetic["mean"]) - float(real["mean"])) / real_std
        std_ratio_delta = abs(float(np.log(synthetic_std / real_std)))
        autocorrelation_delta = abs(
            float(synthetic["autocorrelation_lag1"]) - float(real["autocorrelation_lag1"])
        )
        result: dict[str, float | bool] = {
            "mean_z_delta": mean_z,
            "std_ratio_delta": std_ratio_delta,
            "p05_delta": abs(float(synthetic["p05"]) - float(real["p05"])) / real_std,
            "p50_delta": abs(float(synthetic["p50"]) - float(real["p50"])) / real_std,
            "p95_delta": abs(float(synthetic["p95"]) - float(real["p95"])) / real_std,
            "autocorrelation_delta": autocorrelation_delta,
            "range_violation_rate": float(synthetic["range_violation_rate"]),
        }
        result["passed"] = bool(
            mean_z <= config.max_mean_z_delta
            and std_ratio_delta <= config.max_std_ratio_delta
            and autocorrelation_delta <= config.max_autocorrelation_delta
            and result["range_violation_rate"] <= 0.01
        )
        curve_results[curve] = result

    correlation_distance = _correlation_distance(
        real_stats.get("correlation_matrix", {}),
        synthetic_stats.get("correlation_matrix", {}),
    )
    matrix_passed = correlation_distance <= config.max_correlation_distance
    passed_curves = sum(bool(item.get("passed")) for item in curve_results.values())
    total_curves = len(curve_results)
    fraction = passed_curves / max(total_curves, 1)
    if fraction >= config.min_passed_curves_fraction and matrix_passed:
        status = "passed"
    elif (
        fraction >= max(0.0, config.min_passed_curves_fraction - 0.2)
        and correlation_distance <= config.max_correlation_distance * 1.5
    ):
        status = "warning"
    else:
        status = "failed"
    return {
        "valid": status != "failed",
        "status": status,
        "config": asdict(config),
        "curve_results": curve_results,
        "matrix_results": {
            "correlation_distance": correlation_distance,
            "passed": matrix_passed,
        },
        "summary": {
            "passed_curves": passed_curves,
            "total_curves": total_curves,
            "passed_curves_fraction": fraction,
        },
    }


def _correlation_distance(real: dict[str, Any], synthetic: dict[str, Any]) -> float:
    distances: list[float] = []
    for row, columns in real.items():
        if row not in synthetic:
            continue
        for column, value in columns.items():
            if column in synthetic[row] and row != column:
                distances.append(abs(float(value) - float(synthetic[row][column])))
    return float(np.mean(distances)) if distances else 1.0
