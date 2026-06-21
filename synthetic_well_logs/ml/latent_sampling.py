from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RealismCondition:
    facies: np.ndarray
    lithology: np.ndarray
    fluid: np.ndarray
    target_curves: list[str]
    depth: np.ndarray
    difficulty: str


class LatentSampler:
    """Gaussian proposal distribution used by constrained rejection sampling."""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        self.covariance = covariance + np.eye(len(self.mean)) * 1e-6

    @classmethod
    def from_npz(cls, path: str) -> LatentSampler:
        payload = np.load(path)
        return cls(payload["mean"], payload["covariance"])

    def sample(
        self,
        condition: RealismCondition,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del condition
        return rng.multivariate_normal(self.mean, self.covariance)
