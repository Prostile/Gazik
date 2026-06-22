from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from synthetic_well_logs.rocks import LITHOLOGY
from synthetic_well_logs.student_answers.schema import LithologyInterval
from synthetic_well_logs.student_answers.truth_adapter import TruthData


def _generalized(label: str) -> str:
    return LITHOLOGY.get(label, label)


def score_lithology(predicted: list[LithologyInterval], truth: TruthData) -> dict[str, Any]:
    labels = np.full(len(truth.depth), "__unclassified__", dtype="U32")
    for interval in predicted:
        mask = (truth.depth >= interval.top) & (truth.depth < interval.base)
        labels[mask] = _generalized(interval.lithology)
    expected = truth.lithology.astype(str)
    accuracy = float(np.mean(labels == expected)) if len(expected) else 0.0
    classes = sorted(set(expected))
    class_f1: list[float] = []
    confusion_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for actual, predicted_label in zip(expected, labels, strict=True):
        confusion_counts[str(actual)][str(predicted_label)] += 1
    for label in classes:
        true_positive = int(np.sum((expected == label) & (labels == label)))
        false_positive = int(np.sum((expected != label) & (labels == label)))
        false_negative = int(np.sum((expected == label) & (labels != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(2 * true_positive / denominator if denominator else 0.0)
    confusion: dict[str, dict[str, float]] = {}
    for actual, row in confusion_counts.items():
        total = sum(row.values())
        confusion[f"truth={actual}"] = {
            f"predicted={label}": count / total for label, count in sorted(row.items())
        }
    targeted = {
        ("siltstone", "sandstone"),
        ("coal", "sandstone"),
        ("anhydrite", "limestone"),
        ("sandstone", "coal"),
    }
    major_confusions = [
        {"truth": actual, "predicted": guessed, "fraction": count / sum(row.values())}
        for actual, row in confusion_counts.items()
        for guessed, count in row.items()
        if actual != guessed and guessed != "__unclassified__" and (actual, guessed) in targeted
    ]
    macro_f1 = float(np.mean(class_f1)) if class_f1 else 0.0
    return {
        "task": "lithology_intervals",
        "lithology_accuracy_by_depth": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "major_confusions": major_confusions,
        "score": 100.0 * (0.6 * accuracy + 0.4 * macro_f1),
    }

