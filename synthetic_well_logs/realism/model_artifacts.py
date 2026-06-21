from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthetic_well_logs.ml.autoencoder import AutoencoderConfig, Conv1dAutoencoder
from synthetic_well_logs.ml.latent_sampling import LatentSampler


@dataclass(slots=True)
class RealismModelArtifact:
    root: Path
    model: Conv1dAutoencoder
    config: AutoencoderConfig
    normalization: dict[str, dict[str, float | str]]
    sampler: LatentSampler

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
        sampler = LatentSampler(latent_stats["mean"], latent_stats["covariance"])
        return cls(directory, model, config, normalization, sampler)
