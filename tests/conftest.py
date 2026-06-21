from __future__ import annotations

from pathlib import Path

import lasio
import numpy as np
import pytest

from synthetic_well_logs.datasets import CurveAliasesConfig, IngestionPipeline
from synthetic_well_logs.ml import train_autoencoder


def write_mini_las(path: Path, *, samples: int = 321, aliases: bool = True) -> Path:
    depth_m = 1000.0 + np.arange(samples) * 0.1
    phase = np.linspace(0, 12, samples)
    gr = 75 + 35 * np.sin(phase) + 4 * np.sin(phase * 7)
    rhob = 2.42 - 0.12 * np.sin(phase) + 0.01 * np.cos(phase * 5)
    nphi = 0.22 + 0.08 * np.sin(phase) + 0.008 * np.sin(phase * 6)
    dt = 82 + 15 * np.sin(phase) + 2 * np.cos(phase * 4)
    rt = np.exp(2.0 + 1.2 * np.sin(phase) + 0.1 * np.cos(phase * 8))
    las = lasio.LASFile()
    las.well.WELL = "MINI_CALIBRATION"
    if aliases:
        las.append_curve("MD", depth_m / 0.3048, unit="ft")
        las.append_curve("GRC", gr, unit="API")
        las.append_curve("RHOZ", rhob * 1000, unit="kg/m3")
        las.append_curve("TNPH", nphi * 100, unit="%")
        las.append_curve("DTC", dt / 0.3048, unit="us/m")
        las.append_curve("ILD", rt, unit="ohm.m")
    else:
        las.append_curve("DEPT", depth_m, unit="m")
        las.append_curve("GR", gr, unit="API")
        las.append_curve("RHOB", rhob, unit="g/cc")
        las.append_curve("NPHI", nphi, unit="v/v")
        las.append_curve("DT", dt, unit="us/ft")
        las.append_curve("RT", rt, unit="ohm.m")
    las.write(str(path), version=2.0)
    return path


@pytest.fixture
def mini_las(tmp_path: Path) -> Path:
    return write_mini_las(tmp_path / "mini.las")


@pytest.fixture
def calibration_dataset(tmp_path: Path, mini_las: Path) -> Path:
    out = tmp_path / "calibration"
    pipeline = IngestionPipeline(
        CurveAliasesConfig.from_file(),
        target_step_m=0.1,
        window_size=32,
        stride=16,
        min_valid_fraction=0.8,
    )
    pipeline.ingest(mini_las, out)
    return out


@pytest.fixture
def tiny_model_artifact(tmp_path: Path, calibration_dataset: Path) -> Path:
    out = tmp_path / "model"
    train_autoencoder(
        calibration_dataset,
        out,
        epochs=1,
        latent_dim=4,
        batch_size=8,
        learning_rate=0.001,
    )
    return out
