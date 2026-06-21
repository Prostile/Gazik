from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from synthetic_well_logs.config import ScenarioConfig


@dataclass(frozen=True, slots=True)
class FaciesInterval:
    top: float
    base: float
    facies: str
    lithology: str
    trend: str


@dataclass(slots=True)
class GroundTruth:
    well_id: str
    depth_unit: str
    depth: np.ndarray
    facies: np.ndarray
    lithology: np.ndarray
    vsh: np.ndarray
    phi: np.ndarray
    sw: np.ndarray
    fluid: np.ndarray
    is_reservoir: np.ndarray
    is_pay: np.ndarray
    bad_hole_mask: np.ndarray
    intervals: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    contacts: dict[str, float | None] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedWell:
    well_id: str
    depth: np.ndarray
    curves: pd.DataFrame
    truth: GroundTruth
    scenario: ScenarioConfig
    validation: dict[str, Any] = field(default_factory=dict)

    def export(self, out_prefix: str | Path) -> dict[str, Path]:
        from synthetic_well_logs.export.exporters import export_generated_well

        return export_generated_well(self, Path(out_prefix))
