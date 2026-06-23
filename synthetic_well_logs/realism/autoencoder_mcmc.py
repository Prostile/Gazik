from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.constraints.scoring import TruthSlice, score_curves_against_truth
from synthetic_well_logs.domain import GroundTruth
from synthetic_well_logs.ml.constrained_search import ConstrainedLatentSearch
from synthetic_well_logs.ml.latent_sampling import build_realism_condition
from synthetic_well_logs.realism.model_artifacts import RealismModelArtifact
from synthetic_well_logs.realism.noop import NoOpRealismEnhancer
from synthetic_well_logs.realism.residuals import high_pass, overlap_weights
from synthetic_well_logs.realism.statistical import StatisticalRealismEnhancer


class AutoencoderMCMCRealismEnhancer:
    """Apply residual texture through MCMC-like constrained latent search.

    This is a condition-aware Metropolis-like search, not full Bayesian MCMC:
    latent proposal -> decode -> residual extraction -> physics score -> accept/reject.
    """

    def __init__(self, model_artifact_path: str | Path | None):
        self.model_artifact_path = Path(model_artifact_path) if model_artifact_path else None
        self.artifact: RealismModelArtifact | None = None
        self.last_report: dict[str, object] = {}

    def _whole_well_fallback(
        self,
        base_curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
        reason: str,
    ) -> pd.DataFrame:
        enhancer = self._fallback_enhancer(scenario)
        self.last_report = {
            "mode": "fallback",
            "fallback": scenario.realism.fallback,
            "reason": reason,
            "sampling_strategy": "condition_aware_metropolis_like",
            "condition_strategy": "condition_label_latent_stats",
            "fallback_reasons": [{"reason": reason}],
        }
        return enhancer.enhance(base_curves, truth, scenario, rng)

    @staticmethod
    def _fallback_enhancer(
        scenario: ScenarioConfig,
    ) -> StatisticalRealismEnhancer | NoOpRealismEnhancer:
        if scenario.realism.fallback == "statistical":
            return StatisticalRealismEnhancer()
        return NoOpRealismEnhancer()

    def enhance(
        self,
        base_curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        try:
            if self.model_artifact_path is None:
                raise FileNotFoundError("realism.model_path is not configured")
            self.artifact = RealismModelArtifact.load(self.model_artifact_path)
        except (OSError, ValueError, RuntimeError) as exc:
            return self._whole_well_fallback(base_curves, truth, scenario, rng, str(exc))

        artifact = self.artifact
        artifact.sampler.min_condition_count = scenario.realism.min_condition_count
        channels = list(artifact.config.channels)
        missing = [curve for curve in channels if curve not in base_curves]
        if missing:
            return self._whole_well_fallback(
                base_curves,
                truth,
                scenario,
                rng,
                f"generated well is missing model channels: {missing}",
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            return self._whole_well_fallback(base_curves, truth, scenario, rng, str(exc))

        size = len(base_curves)
        window_size = artifact.config.window_size
        stride = max(1, window_size // 2)
        weights = overlap_weights(window_size)
        residual_sum = {curve: np.zeros(size, dtype=float) for curve in channels}
        weight_sum = np.zeros(size, dtype=float)
        starts = list(range(0, max(1, size - window_size + 1), stride))
        if not starts or starts[-1] != max(0, size - window_size):
            starts.append(max(0, size - window_size))

        accepted = 0
        fallback_windows = 0
        sampling_attempts = 0
        mcmc_steps_total = 0
        accepted_transitions = 0
        condition_usage: Counter[str] = Counter()
        best_scores: list[float] = []
        fallback_reasons: list[dict[str, object]] = []
        fallback_curves: pd.DataFrame | None = None
        searcher = ConstrainedLatentSearch(scenario.realism.mcmc_proposal_scale)

        for start in starts:
            stop = min(size, start + window_size)
            actual = stop - start
            condition = build_realism_condition(
                truth,
                start,
                stop,
                channels,
                scenario.difficulty,
            )
            truth_slice = TruthSlice.from_ground_truth(truth, start, stop)

            def decode_fn(latent: np.ndarray, actual_size: int = actual) -> np.ndarray:
                with torch.no_grad():
                    decoded = artifact.model.decode(
                        torch.from_numpy(latent.astype(np.float32)).unsqueeze(0)
                    )[0].numpy()
                return high_pass(decoded)[:, :actual_size]

            def candidate_from_texture(
                texture: np.ndarray,
                start_index: int = start,
                stop_index: int = stop,
            ) -> dict[str, np.ndarray]:
                candidate: dict[str, np.ndarray] = {}
                for channel, curve in enumerate(channels):
                    base = base_curves[curve].to_numpy(dtype=float)[start_index:stop_index]
                    std = float(artifact.normalization[curve]["std"])
                    if curve == "RT":
                        residual = texture[channel] * std * scenario.realism.strength * 0.25
                        candidate[curve] = base * np.exp(residual)
                    else:
                        residual = texture[channel] * std * scenario.realism.strength * 0.35
                        candidate[curve] = base + residual
                return candidate

            def score_fn(
                texture: np.ndarray,
                current_truth: TruthSlice = truth_slice,
            ) -> float:
                return score_curves_against_truth(
                    candidate_from_texture(texture),
                    current_truth,
                    resistivity_config=scenario.petrophysics.resistivity_model,
                ).total

            selected: dict[str, np.ndarray] | None = None
            window_best_score = float("inf")
            selected_label = "global"
            for attempt in range(scenario.realism.max_attempts_per_window):
                sampling_attempts += 1
                initial = artifact.sampler.sample(condition, rng)
                if attempt == 0:
                    selected_label = artifact.sampler.last_selected_label
                    condition_usage[selected_label] += 1
                search = searcher.search(
                    initial,
                    condition,
                    decode_fn,
                    score_fn,
                    rng,
                    max_steps=scenario.realism.mcmc_steps_per_window,
                    temperature=scenario.realism.mcmc_temperature,
                )
                mcmc_steps_total += search.steps
                accepted_transitions += search.accepted_transitions
                window_best_score = min(window_best_score, search.best_score)
                if search.best_score < scenario.realism.max_constraint_score:
                    selected = candidate_from_texture(decode_fn(search.best_z))
                    accepted += 1
                    break
            best_scores.append(window_best_score)

            if selected is None:
                fallback_windows += 1
                fallback_reasons.append(
                    {
                        "window_start": start,
                        "window_stop": stop,
                        "condition_label": selected_label,
                        "best_score": window_best_score,
                        "reason": "score_above_threshold",
                    }
                )
                if fallback_curves is None:
                    fallback_curves = self._fallback_enhancer(scenario).enhance(
                        base_curves,
                        truth,
                        scenario,
                        rng,
                    )
                selected = {
                    curve: fallback_curves[curve].to_numpy(dtype=float)[start:stop]
                    for curve in channels
                }

            local_weights = weights[:actual]
            for curve in channels:
                base = base_curves[curve].to_numpy(dtype=float)[start:stop]
                if curve == "RT":
                    residual = np.log(
                        np.clip(selected[curve], 1e-12, None) / np.clip(base, 1e-12, None)
                    )
                else:
                    residual = selected[curve] - base
                residual_sum[curve][start:stop] += residual * local_weights
            weight_sum[start:stop] += local_weights

        result = base_curves.copy(deep=True)
        safe_weights = np.maximum(weight_sum, 1e-12)
        residual_summary: dict[str, dict[str, float]] = {}
        for curve in channels:
            residual = residual_sum[curve] / safe_weights
            if curve == "RT":
                result[curve] = base_curves[curve].to_numpy(dtype=float) * np.exp(residual)
                residual_summary[curve] = {
                    "mean_abs_log": float(np.mean(np.abs(residual))),
                    "p95_abs_log": float(np.quantile(np.abs(residual), 0.95)),
                }
            else:
                result[curve] = base_curves[curve].to_numpy(dtype=float) + residual
                residual_summary[curve] = {
                    "mean_abs": float(np.mean(np.abs(residual))),
                    "p95_abs": float(np.quantile(np.abs(residual), 0.95)),
                }
        result["DEPT"] = base_curves["DEPT"].to_numpy(copy=True)
        scores = np.asarray(best_scores, dtype=float)
        self.last_report = {
            "mode": "autoencoder_mcmc",
            "model_path": str(self.model_artifact_path),
            "sampling_strategy": "condition_aware_metropolis_like",
            "condition_strategy": "condition_label_latent_stats",
            "windows": len(starts),
            "accepted_windows": accepted,
            "rejected_windows": fallback_windows,
            "fallback_windows": fallback_windows,
            "sampling_attempts": sampling_attempts,
            "mcmc_steps_total": mcmc_steps_total,
            "accepted_transitions": accepted_transitions,
            "condition_usage": dict(condition_usage),
            "score_summary": {
                "mean_best_score": float(np.mean(scores)),
                "p50_best_score": float(np.quantile(scores, 0.50)),
                "p95_best_score": float(np.quantile(scores, 0.95)),
                "max_best_score": float(np.max(scores)),
            },
            "fallback_reasons": fallback_reasons,
            "residual_summary": residual_summary,
        }
        return result
