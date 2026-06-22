from __future__ import annotations

from typing import Any

from synthetic_well_logs.student_answers.interval_matching import match_intervals
from synthetic_well_logs.student_answers.schema import StudentInterval
from synthetic_well_logs.student_answers.truth_adapter import TruthData


def score_reservoirs(predicted: list[StudentInterval], truth: TruthData) -> dict[str, Any]:
    truth_intervals = truth.mask_intervals(truth.is_reservoir)
    matches = match_intervals(predicted, truth_intervals)
    forbidden = []
    for index, interval in enumerate(predicted):
        mask = (truth.depth >= interval.top) & (truth.depth < interval.base)
        if mask.any() and any(item in {"coal", "anhydrite"} for item in truth.facies[mask]):
            forbidden.append(index)
    return {
        "task": "reservoir_intervals",
        "precision": matches.precision,
        "recall": matches.recall,
        "f1": matches.f1,
        "mean_top_error_m": matches.mean_top_error,
        "mean_base_error_m": matches.mean_base_error,
        "false_positive_intervals": matches.false_positive_indices,
        "missed_intervals": [truth_intervals[index] for index in matches.missed_indices],
        "forbidden_lithology_predictions": forbidden,
        "score": max(0.0, 100.0 * matches.f1 - 15.0 * len(forbidden)),
    }

