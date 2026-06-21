from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from synthetic_well_logs.constraints.scoring import (
    ConstraintScore,
    TruthSlice,
    score_curves_against_truth,
)


def candidate_constraint_score(
    curves: Mapping[str, np.ndarray],
    truth_slice: TruthSlice,
) -> float:
    return score_curves_against_truth(curves, truth_slice).total


def candidate_constraint_report(
    curves: Mapping[str, np.ndarray],
    truth_slice: TruthSlice,
) -> ConstraintScore:
    return score_curves_against_truth(curves, truth_slice)
