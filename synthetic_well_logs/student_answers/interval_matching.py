from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _bounds(interval: Any) -> tuple[float, float]:
    if isinstance(interval, dict):
        return float(interval["top"]), float(interval["base"])
    return float(interval.top), float(interval.base)


def interval_overlap_ratio(
    predicted_top: float,
    predicted_base: float,
    truth_top: float,
    truth_base: float,
) -> float:
    overlap = max(0.0, min(predicted_base, truth_base) - max(predicted_top, truth_top))
    union = max(predicted_base, truth_base) - min(predicted_top, truth_top)
    return overlap / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class IntervalMatch:
    predicted_index: int
    truth_index: int
    overlap_ratio: float
    top_error: float
    base_error: float


@dataclass(frozen=True, slots=True)
class IntervalMatchReport:
    matches: list[IntervalMatch]
    false_positive_indices: list[int]
    missed_indices: list[int]
    precision: float
    recall: float
    f1: float
    mean_top_error: float | None
    mean_base_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "matches": [asdict(item) for item in self.matches],
        }


def match_intervals(
    predicted: list[Any],
    truth: list[Any],
    min_overlap: float = 0.5,
) -> IntervalMatchReport:
    candidates: list[tuple[float, int, int]] = []
    for predicted_index, predicted_interval in enumerate(predicted):
        p_top, p_base = _bounds(predicted_interval)
        for truth_index, truth_interval in enumerate(truth):
            t_top, t_base = _bounds(truth_interval)
            overlap = interval_overlap_ratio(p_top, p_base, t_top, t_base)
            if overlap >= min_overlap:
                candidates.append((overlap, predicted_index, truth_index))
    candidates.sort(reverse=True)
    matched_predicted: set[int] = set()
    matched_truth: set[int] = set()
    matches: list[IntervalMatch] = []
    for overlap, predicted_index, truth_index in candidates:
        if predicted_index in matched_predicted or truth_index in matched_truth:
            continue
        p_top, p_base = _bounds(predicted[predicted_index])
        t_top, t_base = _bounds(truth[truth_index])
        matches.append(
            IntervalMatch(
                predicted_index=predicted_index,
                truth_index=truth_index,
                overlap_ratio=overlap,
                top_error=abs(p_top - t_top),
                base_error=abs(p_base - t_base),
            )
        )
        matched_predicted.add(predicted_index)
        matched_truth.add(truth_index)
    if not predicted and not truth:
        precision = recall = f1 = 1.0
    else:
        precision = len(matches) / len(predicted) if predicted else 0.0
        recall = len(matches) / len(truth) if truth else (1.0 if not predicted else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return IntervalMatchReport(
        matches=matches,
        false_positive_indices=sorted(set(range(len(predicted))) - matched_predicted),
        missed_indices=sorted(set(range(len(truth))) - matched_truth),
        precision=precision,
        recall=recall,
        f1=f1,
        mean_top_error=(
            sum(item.top_error for item in matches) / len(matches) if matches else None
        ),
        mean_base_error=(
            sum(item.base_error for item in matches) / len(matches) if matches else None
        ),
    )
