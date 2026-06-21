from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthetic_well_logs.datasets.parquet_store import read_parquet, write_parquet
from synthetic_well_logs.datasets.windowing import CalibrationWindow


def _transform(curve: str, values: np.ndarray) -> np.ndarray:
    if curve == "RT":
        return np.log(np.clip(values, 1e-6, None))
    return values


def inverse_transform(curve: str, values: np.ndarray) -> np.ndarray:
    if curve == "RT":
        return np.exp(values)
    return values


@dataclass(slots=True)
class CalibrationDataset:
    root: Path
    manifest: dict[str, Any]
    metadata: pd.DataFrame

    @classmethod
    def open(cls, root: str | Path) -> CalibrationDataset:
        directory = Path(root)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        metadata = read_parquet(directory / "metadata.parquet")
        return cls(directory, manifest, metadata)

    @property
    def curve_names(self) -> list[str]:
        return list(self.manifest["curve_names"])

    @property
    def normalization(self) -> dict[str, dict[str, float | str]]:
        return self.manifest["normalization"]

    def load_window(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        row = self.metadata.iloc[index]
        payload = np.load(self.root / row["tensor_path"], allow_pickle=False)
        return payload["curves"].astype(np.float32), payload["valid_mask"].astype(bool)

    def load_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self.metadata.empty:
            shape = (0, len(self.curve_names), int(self.manifest["window_size"]))
            return np.empty(shape, dtype=np.float32), np.empty(shape, dtype=bool)
        windows, masks = zip(
            *(self.load_window(index) for index in range(len(self.metadata))),
            strict=True,
        )
        return np.stack(windows), np.stack(masks)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        output = np.empty_like(values, dtype=float)
        for channel, curve in enumerate(self.curve_names):
            config = self.normalization[curve]
            transformed = values[..., channel, :] * float(config["std"]) + float(config["mean"])
            output[..., channel, :] = inverse_transform(curve, transformed)
        return output


class CalibrationDatasetBuilder:
    def build(
        self,
        windows: list[CalibrationWindow],
        out_dir: str | Path,
        *,
        dataset_id: str,
        source: str,
        target_step_m: float,
        stride: int,
        qc_summary: dict[str, int],
    ) -> CalibrationDataset:
        if not windows:
            raise ValueError("no calibration windows were accepted")
        root = Path(out_dir)
        window_dir = root / "windows"
        window_dir.mkdir(parents=True, exist_ok=True)
        curve_names = windows[0].curve_names
        normalization = self._fit_normalization(windows, curve_names)
        rows: list[dict[str, object]] = []

        for sequence, window in enumerate(windows):
            normalized = self._normalize_window(window, normalization)
            relative = Path("windows") / f"{sequence:06d}.npz"
            np.savez_compressed(
                root / relative,
                curves=normalized.astype(np.float32),
                valid_mask=window.valid_mask.astype(bool),
                depth=window.depth.astype(float),
            )
            rows.append(
                {
                    "window_id": window.window_id,
                    "well_id": window.well_id,
                    "top": window.top,
                    "base": window.base,
                    "source_file": window.source_file,
                    "tensor_path": relative.as_posix(),
                    "valid_fraction": float(window.valid_mask.mean()),
                }
            )

        metadata = pd.DataFrame(rows)
        write_parquet(metadata, root / "metadata.parquet")
        manifest: dict[str, Any] = {
            "dataset_id": dataset_id,
            "created_at": datetime.now(UTC).isoformat(),
            "source": source,
            "curve_names": curve_names,
            "window_size": int(windows[0].curves.shape[1]),
            "stride": stride,
            "target_step_m": target_step_m,
            "well_count": int(metadata["well_id"].nunique()),
            "window_count": len(windows),
            "normalization": normalization,
            "qc_summary": qc_summary,
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return CalibrationDataset(root=root, manifest=manifest, metadata=metadata)

    @staticmethod
    def _fit_normalization(
        windows: list[CalibrationWindow], curve_names: list[str]
    ) -> dict[str, dict[str, float | str]]:
        output: dict[str, dict[str, float | str]] = {}
        for channel, curve in enumerate(curve_names):
            chunks = [
                window.curves[channel][window.valid_mask[channel]]
                for window in windows
                if window.valid_mask[channel].any()
            ]
            values = _transform(curve, np.concatenate(chunks))
            std = max(float(np.std(values)), 1e-6)
            output[curve] = {
                "mean": float(np.mean(values)),
                "std": std,
                "transform": "log" if curve == "RT" else "identity",
            }
        return output

    @staticmethod
    def _normalize_window(
        window: CalibrationWindow,
        normalization: dict[str, dict[str, float | str]],
    ) -> np.ndarray:
        output = np.empty_like(window.curves, dtype=float)
        for channel, curve in enumerate(window.curve_names):
            values = _transform(curve, window.curves[channel])
            config = normalization[curve]
            output[channel] = (values - float(config["mean"])) / float(config["std"])
        window.normalization = normalization
        return output
