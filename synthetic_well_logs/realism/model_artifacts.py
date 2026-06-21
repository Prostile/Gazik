from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthetic_well_logs.ml.autoencoder import AutoencoderConfig, Conv1dAutoencoder
from synthetic_well_logs.ml.latent_sampling import (
    ConditionAwareLatentSampler,
    LatentDistribution,
)


@dataclass(slots=True)
class RealismModelArtifact:
    root: Path
    model: Conv1dAutoencoder
    config: AutoencoderConfig
    normalization: dict[str, dict[str, float | str]]
    sampler: ConditionAwareLatentSampler

    @classmethod
    def load(cls, root: str | Path) -> RealismModelArtifact:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Autoencoder support requires the 'hybrid2' extra: pip install -e '.[hybrid2]'"
            ) from exc
        directory = Path(root)
        required = ["model.pt", "config.json", "normalization.json", "latent_stats.npz"]
        missing = [name for name in required if not (directory / name).exists()]
        if missing:
            raise FileNotFoundError(f"model artifact is incomplete: {', '.join(missing)}")
        config = AutoencoderConfig.from_dict(
            json.loads((directory / "config.json").read_text(encoding="utf-8"))
        )
        model = Conv1dAutoencoder(config)
        state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        normalization = json.loads((directory / "normalization.json").read_text(encoding="utf-8"))
        latent_stats = np.load(directory / "latent_stats.npz")
        condition_stats: dict[str, LatentDistribution] = {}
        condition_json = directory / "latent_stats_by_condition.json"
        condition_npz = directory / "latent_stats_by_condition.npz"
        if condition_json.exists() and condition_npz.exists():
            condition_manifest = json.loads(condition_json.read_text(encoding="utf-8"))
            condition_values = np.load(condition_npz)
            for label, item in condition_manifest.items():
                condition_stats[label] = LatentDistribution(
                    mean=condition_values[item["mean_key"]],
                    covariance=condition_values[item["covariance_key"]],
                    count=int(item["count"]),
                )
        sampler = ConditionAwareLatentSampler(
            latent_stats["mean"],
            latent_stats["covariance"],
            condition_stats,
        )
        return cls(directory, model, config, normalization, sampler)
