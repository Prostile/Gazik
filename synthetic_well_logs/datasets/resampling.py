from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_well_logs.datasets.curve_aliases import CanonicalWell


class DepthResampler:
    def __init__(self, target_step_m: float = 0.1, max_interpolation_gap_m: float = 0.5):
        if target_step_m <= 0 or max_interpolation_gap_m <= 0:
            raise ValueError("resampling steps and gaps must be positive")
        self.target_step_m = target_step_m
        self.max_interpolation_gap_m = max_interpolation_gap_m

    def resample(self, well: CanonicalWell) -> CanonicalWell:
        frame = well.curves.sort_values("DEPT").drop_duplicates("DEPT", keep="first")
        depth = frame["DEPT"].to_numpy(dtype=float)
        finite_depth = np.isfinite(depth)
        frame = frame.loc[finite_depth].reset_index(drop=True)
        depth = frame["DEPT"].to_numpy(dtype=float)
        if depth.size < 2 or not np.all(np.diff(depth) > 0):
            raise ValueError("depth must contain at least two strictly increasing values")

        count = int(np.floor((depth[-1] - depth[0]) / self.target_step_m)) + 1
        target = np.round(depth[0] + np.arange(count) * self.target_step_m, 8)
        output = pd.DataFrame({"DEPT": target})
        for curve in frame.columns:
            if curve == "DEPT":
                continue
            output[curve] = self._interpolate_segments(
                depth,
                frame[curve].to_numpy(dtype=float),
                target,
            )
        well.curves = output
        well.units["DEPT"] = "m"
        return well

    def _interpolate_segments(
        self,
        depth: np.ndarray,
        values: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        output = np.full(target.size, np.nan)
        valid_indices = np.flatnonzero(np.isfinite(values))
        if valid_indices.size < 2:
            return output
        split_at = np.flatnonzero(np.diff(depth[valid_indices]) > self.max_interpolation_gap_m) + 1
        for indices in np.split(valid_indices, split_at):
            if indices.size < 2:
                continue
            mask = (target >= depth[indices[0]]) & (target <= depth[indices[-1]])
            output[mask] = np.interp(target[mask], depth[indices], values[indices])
        return output
