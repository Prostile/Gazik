from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthetic_well_logs.domain import GroundTruth


def _dominant(values: np.ndarray) -> str:
    unique, counts = np.unique(values.astype(str), return_counts=True)
    return str(unique[np.argmax(counts)]) if unique.size else "unknown"


@dataclass(frozen=True, slots=True)
class RealismCondition:
    facies: np.ndarray
    lithology: np.ndarray
    fluid: np.ndarray
    target_curves: list[str]
    depth: np.ndarray
    difficulty: str
    dominant_facies: str
    dominant_lithology: str
    dominant_fluid: str
    facies_fractions: dict[str, float]
    has_pay: bool
    has_gas: bool


def build_realism_condition(
    truth: GroundTruth,
    start: int,
    stop: int,
    target_curves: list[str],
    difficulty: str,
) -> RealismCondition:
    facies = truth.facies[start:stop]
    lithology = truth.lithology[start:stop]
    fluid = truth.fluid[start:stop]
    unique, counts = np.unique(facies.astype(str), return_counts=True)
    fractions = {
        str(name): float(count / max(len(facies), 1))
        for name, count in zip(unique, counts, strict=True)
    }
    return RealismCondition(
        facies=facies,
        lithology=lithology,
        fluid=fluid,
        target_curves=list(target_curves),
        depth=truth.depth[start:stop],
        difficulty=difficulty,
        dominant_facies=_dominant(facies),
        dominant_lithology=_dominant(lithology),
        dominant_fluid=_dominant(fluid),
        facies_fractions=fractions,
        has_pay=bool(np.any(truth.is_pay[start:stop])),
        has_gas=bool(np.any(fluid == "gas")),
    )


def condition_label_from_truth(condition: RealismCondition) -> str:
    if condition.has_pay and condition.dominant_fluid in {"gas", "oil", "mixed"}:
        return "resistive_pay_like"
    if condition.dominant_facies == "shale":
        return "high_gr_shale_like"
    if condition.dominant_facies == "clean_sandstone":
        return "low_gr_clean_sand_like"
    if condition.dominant_facies == "shaly_sandstone":
        return "mixed_shaly_sand_like"
    if condition.dominant_facies in {"tight_sandstone", "limestone", "dolomite"}:
        return "dense_low_porosity_like"
    return "unknown_mixed"


@dataclass(frozen=True, slots=True)
class LatentDistribution:
    mean: np.ndarray
    covariance: np.ndarray
    count: int


class ConditionAwareLatentSampler:
    """Select a geological condition distribution, falling back to global latent stats."""

    def __init__(
        self,
        global_mean: np.ndarray,
        global_covariance: np.ndarray,
        condition_stats: dict[str, LatentDistribution] | None = None,
        min_condition_count: int = 30,
    ):
        self.global_distribution = self._regularize(global_mean, global_covariance, 0)
        self.condition_stats = {
            label: self._regularize(item.mean, item.covariance, item.count)
            for label, item in (condition_stats or {}).items()
        }
        self.min_condition_count = min_condition_count
        self.last_selected_label = "global"

    @staticmethod
    def _regularize(
        mean: np.ndarray,
        covariance: np.ndarray,
        count: int,
    ) -> LatentDistribution:
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim == 0:
            covariance = np.eye(len(mean)) * max(float(covariance), 1e-3)
        covariance = covariance + np.eye(len(mean)) * 1e-6
        return LatentDistribution(mean, covariance, count)

    def distribution_for(self, condition: RealismCondition) -> tuple[str, LatentDistribution]:
        label = condition_label_from_truth(condition)
        distribution = self.condition_stats.get(label)
        if distribution is not None and distribution.count >= self.min_condition_count:
            return label, distribution
        return "global", self.global_distribution

    def sample(
        self,
        condition: RealismCondition,
        rng: np.random.Generator,
    ) -> np.ndarray:
        label, distribution = self.distribution_for(condition)
        self.last_selected_label = label
        proposal = rng.multivariate_normal(distribution.mean, distribution.covariance)
        difficulty_jitter = {"beginner": 0.005, "intermediate": 0.015, "advanced": 0.03}
        channel_factor = max(0.5, len(condition.target_curves) / 5)
        jitter = difficulty_jitter.get(condition.difficulty, 0.015) * channel_factor
        return proposal + rng.normal(0.0, jitter, size=proposal.shape)


LatentSampler = ConditionAwareLatentSampler
