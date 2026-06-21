from __future__ import annotations

from typing import Protocol

import numpy as np

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import FaciesInterval


class IFaciesGenerator(Protocol):
    def generate(
        self,
        depth: np.ndarray,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[FaciesInterval], np.ndarray]: ...
