from __future__ import annotations

import numpy as np

from synthetic_well_logs.config import ResistivityModelConfig
from synthetic_well_logs.domain import GroundTruth
from synthetic_well_logs.forward.model import (
    FLUID_DENSITY,
    FLUID_DT,
    calculate_expected_curves,
)
from synthetic_well_logs.rocks import MATRIX_DENSITY, MATRIX_DT


def expected_curves(
    truth: GroundTruth,
    resistivity_config: ResistivityModelConfig | None = None,
) -> dict[str, np.ndarray]:
    return calculate_expected_curves(truth, resistivity_config)


__all__ = [
    "FLUID_DENSITY",
    "FLUID_DT",
    "MATRIX_DENSITY",
    "MATRIX_DT",
    "expected_curves",
]
