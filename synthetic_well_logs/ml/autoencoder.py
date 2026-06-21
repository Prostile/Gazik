from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as functional
except ImportError:  # pragma: no cover - exercised by optional dependency users
    torch = None
    nn = None
    functional = None


@dataclass(frozen=True, slots=True)
class AutoencoderConfig:
    channels: tuple[str, ...] = ("GR", "RHOB", "NPHI", "DT", "RT")
    window_size: int = 128
    latent_dim: int = 32
    hidden_channels: int = 64

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channels"] = list(self.channels)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AutoencoderConfig:
        return cls(
            channels=tuple(payload["channels"]),
            window_size=int(payload["window_size"]),
            latent_dim=int(payload["latent_dim"]),
            hidden_channels=int(payload.get("hidden_channels", 64)),
        )


class Conv1dAutoencoder(nn.Module if nn is not None else object):
    def __init__(self, config: AutoencoderConfig):
        if nn is None or torch is None or functional is None:
            raise RuntimeError(
                "Autoencoder support requires the 'hybrid2' extra: pip install -e '.[hybrid2]'"
            )
        super().__init__()
        self.config = config
        channels = len(config.channels)
        hidden = config.hidden_channels
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, channels, config.window_size)
            encoded = self.encoder_conv(probe)
        self.encoded_shape = tuple(encoded.shape[1:])
        flattened = int(encoded.numel())
        self.to_latent = nn.Linear(flattened, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, flattened)
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(128, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, channels, kernel_size=5, padding=2),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_conv(inputs)
        return self.to_latent(encoded.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.from_latent(latent).view(latent.shape[0], *self.encoded_shape)
        decoded = functional.interpolate(
            decoded,
            size=self.config.window_size,
            mode="linear",
            align_corners=False,
        )
        return self.decoder_conv(decoded)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))
