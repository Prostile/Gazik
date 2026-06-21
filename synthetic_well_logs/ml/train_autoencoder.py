from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_well_logs.datasets import CalibrationDataset
from synthetic_well_logs.ml.autoencoder import AutoencoderConfig, Conv1dAutoencoder


def _torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Autoencoder support requires the 'hybrid2' extra: pip install -e '.[hybrid2]'"
        ) from exc
    return torch, nn, (DataLoader, TensorDataset)


def train_autoencoder(
    dataset_path: str | Path,
    out_dir: str | Path,
    *,
    epochs: int = 20,
    latent_dim: int = 32,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch, nn, loaders = _torch_modules()
    DataLoader, TensorDataset = loaders
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = CalibrationDataset.open(dataset_path)
    arrays, masks = dataset.load_arrays()
    if not len(arrays):
        raise ValueError("calibration dataset contains no windows")

    config = AutoencoderConfig(
        channels=tuple(dataset.curve_names),
        window_size=int(dataset.manifest["window_size"]),
        latent_dim=latent_dim,
    )
    selected_device = (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else "cpu"
        if device == "auto"
        else device
    )
    model = Conv1dAutoencoder(config).to(selected_device)
    split = max(1, int(len(arrays) * 0.8))
    if split == len(arrays) and len(arrays) > 1:
        split -= 1
    train_x, val_x = arrays[:split], arrays[split:]
    train_m, val_m = masks[:split], masks[split:]
    if not len(val_x):
        val_x, val_m = train_x[-1:], train_m[-1:]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_m)),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_m)),
        batch_size=min(batch_size, len(val_x)),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_losses: list[float] = []
    val_losses: list[float] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        batch_losses: list[float] = []
        for inputs, valid in train_loader:
            inputs = inputs.to(selected_device)
            valid = valid.to(selected_device)
            optimizer.zero_grad()
            reconstructed = model(inputs)
            mse = ((reconstructed - inputs) ** 2)[valid].mean()
            derivative = nn.functional.mse_loss(
                torch.diff(reconstructed, dim=-1),
                torch.diff(inputs, dim=-1),
            )
            loss = mse + 0.05 * derivative
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_losses.append(float(np.mean(batch_losses)))

        model.eval()
        batch_losses = []
        with torch.no_grad():
            for inputs, valid in val_loader:
                inputs = inputs.to(selected_device)
                valid = valid.to(selected_device)
                reconstructed = model(inputs)
                batch_losses.append(float((((reconstructed - inputs) ** 2)[valid].mean()).cpu()))
        val_loss = float(np.mean(batch_losses))
        val_losses.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), root / "model.pt")
    (root / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "normalization.json").write_text(
        json.dumps(dataset.normalization, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model.eval()
    latent_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays), batch_size):
            batch = torch.from_numpy(arrays[start : start + batch_size]).to(selected_device)
            latent_chunks.append(model.encode(batch).cpu().numpy())
    latent = np.concatenate(latent_chunks)
    covariance = np.cov(latent, rowvar=False)
    if covariance.ndim == 0:
        covariance = np.eye(latent_dim) * 1e-3
    np.savez_compressed(
        root / "latent_stats.npz",
        mean=np.mean(latent, axis=0),
        covariance=covariance,
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_epoch": best_epoch,
        "latent_dim": latent_dim,
        "window_size": config.window_size,
        "channels": list(config.channels),
        "device": selected_device,
        "training_windows": int(len(train_x)),
        "validation_windows": int(len(val_x)),
    }
    (root / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
