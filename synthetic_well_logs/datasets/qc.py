from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QC_RANGES = {
    "GR": (0.0, 250.0),
    "RHOB": (1.5, 3.1),
    "NPHI": (-0.15, 0.8),
    "DT": (40.0, 200.0),
    "RT": (0.1, 10_000.0),
}


@dataclass(slots=True)
class CurveQCMasks:
    valid_mask: np.ndarray
    missing_mask: np.ndarray
    range_violation_mask: np.ndarray
    spike_candidate_mask: np.ndarray


@dataclass(slots=True)
class QCResult:
    masks: dict[str, CurveQCMasks]
    reports: list[dict[str, object]]


class QCMaskBuilder:
    def __init__(self, ranges: dict[str, tuple[float, float]] | None = None):
        self.ranges = ranges or QC_RANGES

    def build(self, frame: pd.DataFrame, well_id: str) -> QCResult:
        masks: dict[str, CurveQCMasks] = {}
        reports: list[dict[str, object]] = []
        for curve in frame.columns:
            if curve == "DEPT" or curve not in self.ranges:
                continue
            values = frame[curve].to_numpy(dtype=float)
            missing = ~np.isfinite(values)
            low, high = self.ranges[curve]
            range_violation = np.isfinite(values) & ((values < low) | (values > high))
            spikes = self._spike_mask(values)
            valid = ~(missing | range_violation)
            masks[curve] = CurveQCMasks(valid, missing, range_violation, spikes)
            reports.append(
                {
                    "well_id": well_id,
                    "curve": curve,
                    "sample_count": int(values.size),
                    "valid_count": int(valid.sum()),
                    "missing_rate": float(missing.mean()),
                    "range_violation_rate": float(range_violation.mean()),
                    "spike_candidate_count": int(spikes.sum()),
                }
            )
        return QCResult(masks=masks, reports=reports)

    @staticmethod
    def _spike_mask(values: np.ndarray) -> np.ndarray:
        output = np.zeros(values.size, dtype=bool)
        if values.size < 3:
            return output
        differences = np.diff(values)
        finite = differences[np.isfinite(differences)]
        if finite.size < 3:
            return output
        median = np.median(finite)
        mad = np.median(np.abs(finite - median))
        threshold = max(8 * 1.4826 * mad, np.finfo(float).eps)
        candidates = np.flatnonzero(np.abs(differences - median) > threshold) + 1
        output[candidates] = True
        return output
