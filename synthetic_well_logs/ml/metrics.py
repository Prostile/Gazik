from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from synthetic_well_logs.config import ResistivityModelConfig
from synthetic_well_logs.constraints.scoring import (
    ConstraintScore,
    TruthSlice,
    score_curves_against_truth,
)


def candidate_constraint_score(
    curves: Mapping[str, np.ndarray],
    truth_slice: TruthSlice,
    resistivity_config: ResistivityModelConfig | None = None,
) -> float:
    return score_curves_against_truth(
        curves,
        truth_slice,
        resistivity_config=resistivity_config,
    ).total


def candidate_constraint_report(
    curves: Mapping[str, np.ndarray],
    truth_slice: TruthSlice,
    resistivity_config: ResistivityModelConfig | None = None,
) -> ConstraintScore:
    return score_curves_against_truth(
        curves,
        truth_slice,
        resistivity_config=resistivity_config,
    )
