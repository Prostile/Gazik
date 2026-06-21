from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import GroundTruth


class IForwardLogModel(Protocol):
    def generate(
        self,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame: ...
