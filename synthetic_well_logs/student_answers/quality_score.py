from __future__ import annotations

from typing import Any

from synthetic_well_logs.student_answers.interval_matching import match_intervals
from synthetic_well_logs.student_answers.schema import BadHoleInterval
from synthetic_well_logs.student_answers.truth_adapter import TruthData


def score_quality(predicted: list[BadHoleInterval], truth: TruthData) -> dict[str, Any]:
    truth_intervals = truth.artifact_intervals()
    if not truth_intervals:
        truth_intervals = [
            {**interval, "reason": "washout"}
            for interval in truth.mask_intervals(truth.bad_hole_mask)
        ]
    matches = match_intervals(predicted, truth_intervals, min_overlap=0.35)
    correct_types = sum(
        predicted[item.predicted_index].reason == truth_intervals[item.truth_index]["reason"]
        for item in matches.matches
    )
    type_accuracy = correct_types / len(matches.matches) if matches.matches else 0.0
    if not truth_intervals:
        type_accuracy = 1.0 if not predicted else 0.0
    return {
        "task": "bad_hole_intervals",
        "bad_hole_precision": matches.precision,
        "bad_hole_recall": matches.recall,
        "bad_hole_f1": matches.f1,
        "artifact_type_accuracy": type_accuracy,
        "missed_bad_hole_intervals": [truth_intervals[index] for index in matches.missed_indices],
        "false_bad_hole_intervals": matches.false_positive_indices,
        "score": 100.0 * (0.8 * matches.f1 + 0.2 * type_accuracy),
    }
