import json

import pandas as pd

from synthetic_well_logs.datasets import CalibrationDataset


def test_ingestion_writes_calibration_dataset(calibration_dataset) -> None:
    manifest = json.loads((calibration_dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["well_count"] == 1
    assert manifest["window_count"] > 0
    assert (calibration_dataset / "curves.parquet").exists()
    assert (calibration_dataset / "metadata.parquet").exists()
    assert (calibration_dataset / "ingestion_report.json").exists()
    assert not pd.read_parquet(calibration_dataset / "metadata.parquet").empty

    dataset = CalibrationDataset.open(calibration_dataset)
    arrays, masks = dataset.load_arrays()
    assert arrays.shape[1:] == (5, 32)
    assert arrays.shape == masks.shape
    assert abs(float(arrays.mean())) < 0.5
