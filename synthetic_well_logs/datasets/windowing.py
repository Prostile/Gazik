from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from synthetic_well_logs.datasets.conditions import (
    classify_electrofacies,
    window_curve_medians,
)
from synthetic_well_logs.datasets.constants import MODEL_CURVES
from synthetic_well_logs.datasets.qc import QCResult


@dataclass(slots=True)
class CalibrationWindow:
    window_id: str
    well_id: str
    top: float
    base: float
    depth: np.ndarray
    curves: np.ndarray
    curve_names: list[str]
    valid_mask: np.ndarray
    normalization: dict[str, dict[str, float | str]] = field(default_factory=dict)
    source_file: str = ""
    condition_label: str = "unknown_mixed"
    dominant_electrofacies: str = "unknown_mixed"
    curve_medians: dict[str, float] = field(default_factory=dict)


class WindowSegmenter:
    def __init__(
        self,
        window_size: int = 128,
        stride: int = 64,
        min_valid_fraction: float = 0.85,
        required_curves: list[str] | None = None,
    ):
        if window_size <= 0 or stride <= 0 or stride > window_size:
            raise ValueError("window_size and stride must be positive; stride <= window_size")
        if not 0 < min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be within (0, 1]")
        self.window_size = window_size
        self.stride = stride
        self.min_valid_fraction = min_valid_fraction
        self.required_curves = required_curves or list(MODEL_CURVES)

    def segment(
        self,
        frame: pd.DataFrame,
        qc: QCResult,
        well_id: str,
        source_file: str,
    ) -> list[CalibrationWindow]:
        if any(curve not in frame for curve in self.required_curves):
            return []
        output: list[CalibrationWindow] = []
        last_start = len(frame) - self.window_size
        for start in range(0, max(0, last_start) + 1, self.stride):
            stop = start + self.window_size
            curves = frame.loc[start : stop - 1, self.required_curves].to_numpy(dtype=float).T
            valid = np.vstack(
                [qc.masks[curve].valid_mask[start:stop] for curve in self.required_curves]
            )
            valid &= np.isfinite(curves)
            if float(valid.mean()) < self.min_valid_fraction:
                continue
            filled = curves.copy()
            for channel in range(filled.shape[0]):
                channel_values = filled[channel]
                fill = float(np.nanmedian(channel_values[valid[channel]]))
                channel_values[~valid[channel]] = fill
            depth = frame["DEPT"].to_numpy(dtype=float)[start:stop]
            medians = window_curve_medians(filled, self.required_curves, valid)
            condition_label = classify_electrofacies(medians)
            output.append(
                CalibrationWindow(
                    window_id=f"{well_id}_{start:08d}",
                    well_id=well_id,
                    top=float(depth[0]),
                    base=float(depth[-1]),
                    depth=depth,
                    curves=filled,
                    curve_names=list(self.required_curves),
                    valid_mask=valid,
                    source_file=source_file,
                    condition_label=condition_label,
                    dominant_electrofacies=condition_label,
                    curve_medians=medians,
                )
            )
        return output
