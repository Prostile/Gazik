from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.constraints.physics_constraints import RANGES
from synthetic_well_logs.domain import GroundTruth

NOISE_SCALE = {"GR": 1.4, "CALI": 0.025, "RHOB": 0.007, "NPHI": 0.006, "DT": 0.8}


class ArtifactSimulator:
    """Apply observable measurement artifacts and record their exact provenance."""

    def apply(
        self,
        curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        result = curves.copy(deep=True)
        if scenario.artifacts.noise:
            self._noise(result, truth, rng)
        if scenario.artifacts.washout:
            self._washout(result, truth, scenario, rng)
        if scenario.artifacts.spikes:
            self._spikes(result, truth, rng)
        if scenario.artifacts.missing_intervals:
            self._missing_interval(result, truth, scenario, rng)
        if scenario.artifacts.depth_shift:
            self._depth_shift(result, truth, scenario, rng)
        for curve, bounds in RANGES.items():
            if curve in result:
                result[curve] = result[curve].clip(*bounds)
        return result

    @staticmethod
    def _noise(
        curves: pd.DataFrame,
        truth: GroundTruth,
        rng: np.random.Generator,
    ) -> None:
        affected: list[str] = []
        for curve in curves.columns:
            if curve == "DEPT":
                continue
            affected.append(curve)
            raw = rng.normal(size=len(curves))
            if curve == "RT":
                curves[curve] *= np.exp(raw * 0.018)
            else:
                curves[curve] += raw * NOISE_SCALE[curve]
        truth.artifacts.append(
            {
                "top": float(truth.depth[0]),
                "base": float(truth.depth[-1]),
                "type": "noise",
                "affected_curves": affected,
            }
        )

    @staticmethod
    def _interval_indices(
        size: int,
        step: float,
        unit: str,
        rng: np.random.Generator,
        min_m: float,
        max_m: float,
    ) -> tuple[int, int]:
        length_m = float(rng.uniform(min_m, max_m))
        length = length_m if unit == "m" else length_m * 3.28084
        count = max(2, int(round(length / step)))
        count = min(count, max(2, size // 5))
        start = int(rng.integers(max(1, size // 12), max(2, size - count - 1)))
        return start, min(size, start + count)

    def _washout(
        self,
        curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> None:
        shale_indices = np.flatnonzero(truth.facies == "shale")
        washout_m = float(rng.uniform(5.0, 11.0))
        washout_length = washout_m if scenario.depth.unit == "m" else washout_m * 3.28084
        count = max(3, int(round(washout_length / scenario.depth.step)))
        if shale_indices.size >= count:
            candidates = shale_indices[:-count] if shale_indices.size > count else shale_indices
            candidate = int(rng.choice(candidates))
            start = min(candidate, len(curves) - count)
            stop = start + count
        else:
            start, stop = self._interval_indices(
                len(curves), scenario.depth.step, scenario.depth.unit, rng, 5.0, 11.0
            )
        truth.bad_hole_mask[start:stop] = True
        if "CALI" in curves:
            curves.loc[start : stop - 1, "CALI"] += rng.uniform(2.0, 4.5)
        if "RHOB" in curves:
            curves.loc[start : stop - 1, "RHOB"] -= rng.uniform(0.08, 0.22)
            curves.loc[start : stop - 1, "RHOB"] += rng.normal(0, 0.035, stop - start)
        if "NPHI" in curves:
            curves.loc[start : stop - 1, "NPHI"] += rng.uniform(0.05, 0.13)
            curves.loc[start : stop - 1, "NPHI"] += rng.normal(0, 0.025, stop - start)
        affected = [curve for curve in ("CALI", "RHOB", "NPHI") if curve in curves]
        truth.artifacts.append(self._artifact_record(truth, start, stop, "washout", affected))

    @staticmethod
    def _spikes(
        curves: pd.DataFrame,
        truth: GroundTruth,
        rng: np.random.Generator,
    ) -> None:
        candidates = [curve for curve in ("GR", "RHOB", "NPHI", "DT", "RT") if curve in curves]
        if not candidates:
            return
        count = max(2, min(8, len(curves) // 500 + 2))
        locations = np.sort(rng.choice(len(curves), size=count, replace=False))
        affected: set[str] = set()
        for location in locations:
            curve = str(rng.choice(candidates))
            affected.add(curve)
            sign = float(rng.choice([-1.0, 1.0]))
            if curve == "RT":
                curves.at[int(location), curve] *= rng.uniform(2.5, 7.0)
            else:
                scale = {"GR": 35.0, "RHOB": 0.18, "NPHI": 0.12, "DT": 18.0}[curve]
                curves.at[int(location), curve] += sign * scale * rng.uniform(0.7, 1.3)
        truth.artifacts.append(
            {
                "top": float(truth.depth[locations[0]]),
                "base": float(truth.depth[locations[-1]]),
                "type": "spikes",
                "affected_curves": sorted(affected),
                "sample_depths": [round(float(truth.depth[index]), 6) for index in locations],
            }
        )

    def _missing_interval(
        self,
        curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> None:
        candidates = [curve for curve in ("NPHI", "DT", "RHOB", "RT") if curve in curves]
        if not candidates:
            return
        curve = str(rng.choice(candidates))
        start, stop = self._interval_indices(
            len(curves), scenario.depth.step, scenario.depth.unit, rng, 1.0, 3.5
        )
        curves.loc[start : stop - 1, curve] = np.nan
        truth.artifacts.append(
            self._artifact_record(truth, start, stop, "missing_interval", [curve])
        )

    def _depth_shift(
        self,
        curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> None:
        candidates = [curve for curve in ("GR", "RHOB", "NPHI", "DT", "RT") if curve in curves]
        if not candidates:
            return
        curve = str(rng.choice(candidates))
        shift_m = float(rng.uniform(0.2, 2.0))
        shift = shift_m if scenario.depth.unit == "m" else shift_m * 3.28084
        shift_samples = max(1, int(round(shift / scenario.depth.step)))
        values = curves[curve].to_numpy(copy=True)
        shifted = np.full_like(values, np.nan)
        shifted[shift_samples:] = values[:-shift_samples]
        curves[curve] = shifted
        truth.artifacts.append(
            {
                "top": float(truth.depth[0]),
                "base": float(truth.depth[-1]),
                "type": "depth_shift",
                "affected_curves": [curve],
                "shift_m": round(
                    shift_samples
                    * scenario.depth.step
                    * (1.0 if scenario.depth.unit == "m" else 0.3048),
                    6,
                ),
            }
        )

    @staticmethod
    def _artifact_record(
        truth: GroundTruth,
        start: int,
        stop: int,
        artifact_type: str,
        curves: list[str],
    ) -> dict[str, object]:
        return {
            "top": round(float(truth.depth[start]), 6),
            "base": round(float(truth.depth[min(stop - 1, len(truth.depth) - 1)]), 6),
            "type": artifact_type,
            "affected_curves": curves,
        }
