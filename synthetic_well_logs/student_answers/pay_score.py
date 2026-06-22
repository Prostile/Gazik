from __future__ import annotations

from typing import Any

import numpy as np

from synthetic_well_logs.student_answers.interval_matching import match_intervals
from synthetic_well_logs.student_answers.schema import PayInterval
from synthetic_well_logs.student_answers.truth_adapter import TruthData


def score_pay(predicted: list[PayInterval], truth: TruthData) -> dict[str, Any]:
    truth_intervals = truth.mask_intervals(truth.is_pay)
    matches = match_intervals(predicted, truth_intervals)
    fluid_correct = 0
    for match in matches.matches:
        interval = predicted[match.predicted_index]
        t_interval = truth_intervals[match.truth_index]
        mask = (truth.depth >= t_interval["top"]) & (truth.depth < t_interval["base"])
        fluids, counts = np.unique(truth.fluid[mask], return_counts=True)
        expected = str(fluids[np.argmax(counts)])
        fluid_correct += int(interval.fluid == expected)
    fluid_accuracy = fluid_correct / len(matches.matches) if matches.matches else 0.0
    forbidden: list[int] = []
    for index, interval in enumerate(predicted):
        mask = (truth.depth >= interval.top) & (truth.depth < interval.base)
        if mask.any() and np.isin(truth.facies[mask], ["coal", "anhydrite"]).any():
            forbidden.append(index)
    missed_thickness = sum(
        truth_intervals[index]["base"] - truth_intervals[index]["top"]
        for index in matches.missed_indices
    )
    fluid_component = fluid_accuracy if truth_intervals else (1.0 if not predicted else 0.0)
    score = max(0.0, 100.0 * (0.8 * matches.f1 + 0.2 * fluid_component) - 20 * len(forbidden))
    return {
        "task": "pay_intervals",
        "pay_precision": matches.precision,
        "pay_recall": matches.recall,
        "pay_f1": matches.f1,
        "fluid_accuracy": fluid_component,
        "false_pay_from_nonreservoir": matches.false_positive_indices,
        "forbidden_lithology_predictions": forbidden,
        "missed_pay_thickness_m": missed_thickness,
        "score": score,
    }

