from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig


class ToolResolutionSmoother:
    """Approximate finite vertical tool resolution without filling data gaps."""

    def apply(self, curves: pd.DataFrame, scenario: ScenarioConfig) -> pd.DataFrame:
        result = curves.copy(deep=True)
        if not scenario.tool_resolution.enabled:
            return result

        unit_factor = 1.0 if scenario.depth.unit == "m" else 0.3048
        step_m = scenario.depth.step * unit_factor
        for curve, window_m in scenario.tool_resolution.windows_m.items():
            if curve not in result:
                continue
            samples = max(1, int(round(window_m / step_m)))
            if samples <= 1:
                continue
            if samples % 2 == 0:
                samples += 1
            original = result[curve].to_numpy(dtype=float)
            missing = ~np.isfinite(original)
            values = np.log(np.clip(original, 1e-12, None)) if curve == "RT" else original
            smoothed = (
                pd.Series(values)
                .rolling(samples, center=True, min_periods=max(1, samples // 2))
                .mean()
                .to_numpy(copy=True)
            )
            if curve == "RT":
                smoothed = np.exp(smoothed)
            smoothed[missing] = np.nan
            result[curve] = smoothed
        return result
