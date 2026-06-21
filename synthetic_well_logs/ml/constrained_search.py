from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from synthetic_well_logs.ml.latent_sampling import RealismCondition


@dataclass(frozen=True, slots=True)
class LatentSearchResult:
    best_z: np.ndarray
    best_score: float
    final_score: float
    steps: int
    accepted_transitions: int


class ConstrainedLatentSearch:
    """Metropolis-like local search; this is not full Bayesian MCMC."""

    def __init__(self, proposal_scale: float = 0.15):
        if proposal_scale <= 0:
            raise ValueError("proposal_scale must be positive")
        self.proposal_scale = proposal_scale

    def search(
        self,
        initial_z: np.ndarray,
        condition: RealismCondition,
        decode_fn: Callable[[np.ndarray], np.ndarray],
        score_fn: Callable[[np.ndarray], float],
        rng: np.random.Generator,
        max_steps: int,
        temperature: float,
    ) -> LatentSearchResult:
        del condition
        if max_steps < 1 or temperature <= 0:
            raise ValueError("max_steps and temperature must be positive")
        current = np.asarray(initial_z, dtype=float).copy()
        current_score = float(score_fn(decode_fn(current)))
        best = current.copy()
        best_score = current_score
        accepted = 0
        for _ in range(max_steps):
            proposed = current + rng.normal(0.0, self.proposal_scale, size=current.shape)
            proposed_score = float(score_fn(decode_fn(proposed)))
            delta = proposed_score - current_score
            accept_probability = 1.0 if delta <= 0 else float(np.exp(-delta / temperature))
            if rng.random() <= accept_probability:
                current = proposed
                current_score = proposed_score
                accepted += 1
            if proposed_score < best_score:
                best = proposed.copy()
                best_score = proposed_score
        return LatentSearchResult(
            best_z=best,
            best_score=best_score,
            final_score=current_score,
            steps=max_steps,
            accepted_transitions=accepted,
        )
