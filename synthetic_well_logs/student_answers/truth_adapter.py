from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from synthetic_well_logs.domain import GroundTruth


@dataclass(slots=True)
class TruthData:
    well_id: str
    depth: np.ndarray
    facies: np.ndarray
    lithology: np.ndarray
    fluid: np.ndarray
    is_reservoir: np.ndarray
    is_pay: np.ndarray
    bad_hole_mask: np.ndarray
    intervals: list[dict[str, Any]]
    artifacts: list[dict[str, Any]]

    @classmethod
    def from_source(cls, source: GroundTruth | Mapping[str, Any]) -> TruthData:
        if isinstance(source, GroundTruth):
            return cls(
                well_id=source.well_id,
                depth=np.asarray(source.depth, dtype=float),
                facies=np.asarray(source.facies, dtype=str),
                lithology=np.asarray(source.lithology, dtype=str),
                fluid=np.asarray(source.fluid, dtype=str),
                is_reservoir=np.asarray(source.is_reservoir, dtype=bool),
                is_pay=np.asarray(source.is_pay, dtype=bool),
                bad_hole_mask=np.asarray(source.bad_hole_mask, dtype=bool),
                intervals=list(source.intervals),
                artifacts=list(source.artifacts),
            )
        samples = source.get("samples", {})
        required = {
            "depth",
            "facies",
            "lithology",
            "fluid",
            "is_reservoir",
            "is_pay",
            "bad_hole_mask",
        }
        missing = required - set(samples)
        if missing:
            raise ValueError(f"truth JSON is missing sample arrays: {sorted(missing)}")
        return cls(
            well_id=str(source.get("well_id", "")),
            depth=np.asarray(samples["depth"], dtype=float),
            facies=np.asarray(samples["facies"], dtype=str),
            lithology=np.asarray(samples["lithology"], dtype=str),
            fluid=np.asarray(samples["fluid"], dtype=str),
            is_reservoir=np.asarray(samples["is_reservoir"], dtype=bool),
            is_pay=np.asarray(samples["is_pay"], dtype=bool),
            bad_hole_mask=np.asarray(samples["bad_hole_mask"], dtype=bool),
            intervals=list(source.get("intervals", [])),
            artifacts=list(source.get("artifacts", [])),
        )

    @property
    def step(self) -> float:
        return float(np.median(np.diff(self.depth))) if len(self.depth) > 1 else 0.1

    def mask_intervals(self, mask: np.ndarray) -> list[dict[str, float]]:
        values = np.asarray(mask, dtype=bool)
        if not values.any():
            return []
        changes = np.flatnonzero(values[1:] != values[:-1]) + 1
        starts = np.r_[0, changes]
        stops = np.r_[changes, len(values)]
        output: list[dict[str, float]] = []
        for start, stop in zip(starts, stops, strict=True):
            if not values[start]:
                continue
            output.append(
                {
                    "top": float(self.depth[start]),
                    "base": (
                        float(self.depth[stop])
                        if stop < len(values)
                        else float(self.depth[-1] + self.step)
                    ),
                }
            )
        return output

    def artifact_intervals(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for artifact in self.artifacts:
            artifact_type = str(artifact.get("type", "other"))
            if artifact_type == "noise":
                continue
            if artifact_type == "spikes" and artifact.get("sample_depths"):
                output.extend(
                    {
                        "top": float(depth),
                        "base": float(depth) + self.step,
                        "reason": "spikes",
                    }
                    for depth in artifact["sample_depths"]
                )
            else:
                output.append(
                    {
                        "top": float(artifact["top"]),
                        "base": max(float(artifact["base"]), float(artifact["top"]) + self.step),
                        "reason": artifact_type,
                    }
                )
        return output
