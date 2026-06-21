from __future__ import annotations

import numpy as np

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import FaciesInterval

LITHOLOGY = {
    "shale": "shale",
    "clean_sandstone": "sandstone",
    "shaly_sandstone": "sandstone",
    "tight_sandstone": "sandstone",
    "limestone": "limestone",
    "dolomite": "dolomite",
}

THICKNESS_M = {
    "shale": (7.0, 0.55),
    "shaly_sandstone": (5.0, 0.50),
    "clean_sandstone": (7.5, 0.45),
    "tight_sandstone": (4.0, 0.45),
    "limestone": (9.0, 0.50),
    "dolomite": (8.0, 0.50),
}


class SemiMarkovFaciesGenerator:
    """Generate discrete beds with explicit, stochastic bed thicknesses."""

    def generate(
        self,
        depth: np.ndarray,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[FaciesInterval], np.ndarray]:
        facies_set = scenario.geology.facies_set
        values = np.empty(depth.size, dtype="U24")
        current = facies_set[0]
        cursor = 0

        while cursor < depth.size:
            mean_m, coefficient = THICKNESS_M[current]
            shape = 1.0 / coefficient**2
            scale = mean_m / shape
            thickness_m = max(scenario.depth.step, float(rng.gamma(shape, scale)))
            thickness = thickness_m if scenario.depth.unit == "m" else thickness_m * 3.28084
            count = max(1, int(round(thickness / scenario.depth.step)))
            stop = min(depth.size, cursor + count)
            values[cursor:stop] = current
            cursor = stop
            if cursor < depth.size:
                current = self._next_facies(current, facies_set, scenario, rng)

        target_mask = self._inject_learning_target(values, scenario, rng)
        intervals = self._to_intervals(depth, values, target_mask, scenario)
        return values, intervals, target_mask

    @staticmethod
    def _next_facies(
        current: str,
        facies_set: list[str],
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> str:
        weights = np.ones(len(facies_set), dtype=float)
        for index, candidate in enumerate(facies_set):
            if candidate == current:
                weights[index] = 0.15
            if {current, candidate} <= {"shale", "shaly_sandstone", "clean_sandstone"}:
                weights[index] *= 2.0

        pattern = scenario.geology.stacking_pattern.lower()
        if "coarsening" in pattern:
            rank = {
                "shale": 0,
                "shaly_sandstone": 1,
                "clean_sandstone": 2,
                "tight_sandstone": 2,
            }
            current_rank = rank.get(current, 1)
            for index, candidate in enumerate(facies_set):
                if rank.get(candidate, 1) >= current_rank:
                    weights[index] *= 1.5
        elif "alternation" in pattern:
            for index, candidate in enumerate(facies_set):
                if (current == "shale") != (candidate == "shale"):
                    weights[index] *= 1.8

        weights /= weights.sum()
        return str(rng.choice(facies_set, p=weights))

    @staticmethod
    def _inject_learning_target(
        values: np.ndarray,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Guarantee that every educational scenario contains its requested target."""
        low, high = scenario.target.net_pay_thickness_m
        thickness_m = float(rng.uniform(low, high))
        thickness = thickness_m if scenario.depth.unit == "m" else thickness_m * 3.28084
        count = max(2, int(round(thickness / scenario.depth.step)))
        count = min(count, max(2, values.size // 3))
        center = int(values.size * rng.uniform(0.38, 0.68))
        start = max(1, min(values.size - count - 1, center - count // 2))
        values[start : start + count] = scenario.target.reservoir_type
        target_mask = np.zeros(values.size, dtype=bool)
        target_mask[start : start + count] = True
        return target_mask

    @staticmethod
    def _to_intervals(
        depth: np.ndarray,
        values: np.ndarray,
        target_mask: np.ndarray,
        scenario: ScenarioConfig,
    ) -> list[FaciesInterval]:
        changes = (
            np.flatnonzero((values[1:] != values[:-1]) | (target_mask[1:] != target_mask[:-1])) + 1
        )
        starts = np.r_[0, changes]
        stops = np.r_[changes, values.size]
        intervals: list[FaciesInterval] = []
        for start, stop in zip(starts, stops, strict=True):
            base = scenario.depth.stop if stop == values.size else float(depth[stop])
            facies = str(values[start])
            intervals.append(
                FaciesInterval(
                    top=float(depth[start]),
                    base=base,
                    facies=facies,
                    lithology=LITHOLOGY[facies],
                    trend=scenario.geology.stacking_pattern,
                )
            )
        return intervals
