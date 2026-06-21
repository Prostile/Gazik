from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.domain import GroundTruth
from synthetic_well_logs.ml.latent_sampling import RealismCondition
from synthetic_well_logs.ml.metrics import candidate_constraint_score
from synthetic_well_logs.realism.model_artifacts import RealismModelArtifact
from synthetic_well_logs.realism.noop import NoOpRealismEnhancer
from synthetic_well_logs.realism.residuals import high_pass, overlap_weights
from synthetic_well_logs.realism.statistical import StatisticalRealismEnhancer


class AutoencoderMCMCRealismEnhancer:
    """Apply sampled autoencoder texture as overlap-added residuals only."""

    def __init__(self, model_artifact_path: str | Path | None):
        self.model_artifact_path = Path(model_artifact_path) if model_artifact_path else None
        self.artifact: RealismModelArtifact | None = None
        self.last_report: dict[str, object] = {}

    def _fallback(
        self,
        base_curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
        reason: str,
    ) -> pd.DataFrame:
        enhancer = (
            StatisticalRealismEnhancer()
            if scenario.realism.fallback == "statistical"
            else NoOpRealismEnhancer()
        )
        self.last_report = {
            "mode": "fallback",
            "fallback": scenario.realism.fallback,
            "reason": reason,
        }
        return enhancer.enhance(base_curves, truth, scenario, rng)

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
            return self._fallback(base_curves, truth, scenario, rng, str(exc))

        artifact = self.artifact
        channels = list(artifact.config.channels)
        missing = [curve for curve in channels if curve not in base_curves]
        if missing:
            return self._fallback(
                base_curves,
                truth,
                scenario,
                rng,
                f"generated well is missing model channels: {missing}",
            )
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            return self._fallback(base_curves, truth, scenario, rng, str(exc))

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
        attempt_count = 0
        fallback_curves: pd.DataFrame | None = None

        for start in starts:
            stop = min(size, start + window_size)
            actual = stop - start
            condition = RealismCondition(
                facies=truth.facies[start:stop],
                lithology=truth.lithology[start:stop],
                fluid=truth.fluid[start:stop],
                target_curves=channels,
                depth=truth.depth[start:stop],
                difficulty=scenario.difficulty,
            )
            selected: dict[str, np.ndarray] | None = None
            for _ in range(scenario.realism.max_attempts_per_window):
                attempt_count += 1
                latent = artifact.sampler.sample(condition, rng)
                with torch.no_grad():
                    decoded = artifact.model.decode(
                        torch.from_numpy(latent.astype(np.float32)).unsqueeze(0)
                    )[0].numpy()
                texture = high_pass(decoded)[:, :actual]
                candidate: dict[str, np.ndarray] = {}
                for channel, curve in enumerate(channels):
                    base = base_curves[curve].to_numpy(dtype=float)[start:stop]
                    std = float(artifact.normalization[curve]["std"])
                    if curve == "RT":
                        residual = texture[channel] * std * scenario.realism.strength * 0.25
                        candidate[curve] = base * np.exp(residual)
                    else:
                        residual = texture[channel] * std * scenario.realism.strength * 0.35
                        candidate[curve] = base + residual
                truth_slice = {
                    "vsh": truth.vsh[start:stop],
                    "phi": truth.phi[start:stop],
                    "fluid": truth.fluid[start:stop],
                    "is_pay": truth.is_pay[start:stop],
                }
                if (
                    candidate_constraint_score(candidate, truth_slice)
                    <= scenario.realism.max_constraint_score
                ):
                    selected = candidate
                    accepted += 1
                    break

            if selected is None:
                fallback_windows += 1
                if fallback_curves is None:
                    fallback_curves = self._fallback(
                        base_curves,
                        truth,
                        scenario,
                        rng,
                        "constrained sampling rejected one or more windows",
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
        for curve in channels:
            residual = residual_sum[curve] / safe_weights
            if curve == "RT":
                result[curve] = base_curves[curve].to_numpy(dtype=float) * np.exp(residual)
            else:
                result[curve] = base_curves[curve].to_numpy(dtype=float) + residual
        result["DEPT"] = base_curves["DEPT"].to_numpy(copy=True)
        self.last_report = {
            "mode": "autoencoder_mcmc",
            "model_path": str(self.model_artifact_path),
            "windows": len(starts),
            "accepted_windows": accepted,
            "fallback_windows": fallback_windows,
            "sampling_attempts": attempt_count,
        }
        return result
