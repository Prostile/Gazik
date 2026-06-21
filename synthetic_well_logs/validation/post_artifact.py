from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.constraints.physics_constraints import RANGES
from synthetic_well_logs.domain import GroundTruth


class PostArtifactValidator:
    def validate(
        self,
        curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
    ) -> dict[str, Any]:
        expected_nan = {
            curve: np.zeros(len(curves), dtype=bool) for curve in curves if curve != "DEPT"
        }
        artifact_types = {str(item["type"]) for item in truth.artifacts}
        for artifact in truth.artifacts:
            artifact_type = artifact["type"]
            affected = artifact.get("affected_curves", [])
            if artifact_type == "missing_interval":
                mask = (truth.depth >= artifact["top"]) & (truth.depth <= artifact["base"])
                for curve in affected:
                    if curve in expected_nan:
                        expected_nan[curve] |= mask
            elif artifact_type == "depth_shift":
                samples = max(
                    1,
                    int(
                        round(
                            float(artifact["shift_m"])
                            / (
                                scenario.depth.step
                                * (1.0 if scenario.depth.unit == "m" else 0.3048)
                            )
                        )
                    ),
                )
                for curve in affected:
                    if curve in expected_nan:
                        expected_nan[curve][:samples] = True

        unexpected_nan: dict[str, int] = {}
        for curve, expected in expected_nan.items():
            observed = ~np.isfinite(curves[curve].to_numpy(dtype=float))
            unexpected_nan[curve] = int(np.sum(observed & ~expected))
        range_violations = {
            curve: int(np.sum(((values < low) | (values > high)) & np.isfinite(values)))
            for curve, (low, high) in RANGES.items()
            if curve in curves
            for values in [curves[curve].to_numpy(dtype=float)]
        }
        expected_artifacts = {
            name
            for name in ("noise", "washout", "spikes", "missing_intervals", "depth_shift")
            if getattr(scenario.artifacts, name)
        }
        expected_artifacts.discard("missing_intervals")
        if scenario.artifacts.missing_intervals:
            expected_artifacts.add("missing_interval")
        provenance_ok = expected_artifacts <= artifact_types
        washout = truth.bad_hole_mask
        washout_recorded = (not washout.any()) or "washout" in artifact_types
        cali_consistent = True
        if washout.any() and "CALI" in curves:
            cali = curves["CALI"].to_numpy(dtype=float)
            cali_consistent = bool(np.nanmean(cali[washout]) > np.nanmedian(cali[~washout]) + 0.5)
        depth_unchanged = bool(
            "DEPT" in curves and np.array_equal(curves["DEPT"].to_numpy(dtype=float), truth.depth)
        )
        valid = bool(
            not sum(unexpected_nan.values())
            and not sum(range_violations.values())
            and provenance_ok
            and washout_recorded
            and cali_consistent
            and depth_unchanged
            and len(curves) == len(truth.depth)
        )
        return {
            "valid": valid,
            "unexpected_nan_count": unexpected_nan,
            "range_violations": range_violations,
            "artifact_provenance_preserved": provenance_ok,
            "bad_hole_matches_washout": washout_recorded,
            "cali_washout_consistent": cali_consistent,
            "depth_unchanged": depth_unchanged,
            "sample_count_unchanged": len(curves) == len(truth.depth),
        }
