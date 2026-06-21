from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import GroundTruth

CURVE_SCALE = {"GR": 4.5, "CALI": 0.035, "RHOB": 0.025, "NPHI": 0.018, "DT": 2.3}


def _ar1(size: int, rng: np.random.Generator, coefficient: float) -> np.ndarray:
    raw = rng.normal(size=size)
    output = np.empty(size)
    output[0] = raw[0]
    innovation_scale = np.sqrt(1 - coefficient**2)
    for index in range(1, size):
        output[index] = coefficient * output[index - 1] + innovation_scale * raw[index]
    return output


class StatisticalRealismEnhancer:
    """Add constrained, correlated texture without changing geological labels."""

    def enhance(
        self,
        base_curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        result = base_curves.copy(deep=True)
        strength = scenario.realism.strength
        if strength == 0:
            return result

        shared = _ar1(len(result), rng, 0.92)
        for curve in result.columns:
            if curve == "DEPT":
                continue
            local = _ar1(len(result), rng, 0.78)
            texture = 0.62 * shared + 0.78 * local
            facies_factor = np.where(truth.facies == "shale", 1.25, 1.0)
            if curve == "RT":
                result[curve] *= np.exp(texture * 0.09 * strength * facies_factor)
            else:
                scale = CURVE_SCALE[curve] * strength * facies_factor
                result[curve] += texture * scale
        return result

