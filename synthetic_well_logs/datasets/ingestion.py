from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lasio
import pandas as pd

from synthetic_well_logs.datasets.calibration_dataset import (
    CalibrationDataset,
    CalibrationDatasetBuilder,
)
from synthetic_well_logs.datasets.constants import MODEL_CURVES
from synthetic_well_logs.datasets.curve_aliases import CurveAliasesConfig, CurveCanonicalizer
from synthetic_well_logs.datasets.parquet_store import write_parquet
from synthetic_well_logs.datasets.qc import QCMaskBuilder
from synthetic_well_logs.datasets.reports import curve_statistics
from synthetic_well_logs.datasets.resampling import DepthResampler
from synthetic_well_logs.datasets.units import UnitNormalizer
from synthetic_well_logs.datasets.windowing import CalibrationWindow, WindowSegmenter


@dataclass(slots=True)
class IngestionResult:
    dataset: CalibrationDataset
    report: dict[str, Any]


class IngestionPipeline:
    def __init__(
        self,
        aliases: CurveAliasesConfig,
        *,
        target_step_m: float = 0.1,
        max_interpolation_gap_m: float = 0.5,
        window_size: int = 128,
        stride: int = 64,
        min_valid_fraction: float = 0.85,
        required_model_curves: tuple[str, ...] = MODEL_CURVES,
    ):
        self.canonicalizer = CurveCanonicalizer(aliases)
        self.normalizer = UnitNormalizer()
        self.resampler = DepthResampler(target_step_m, max_interpolation_gap_m)
        self.qc = QCMaskBuilder()
        self.required_model_curves = required_model_curves
        self.segmenter = WindowSegmenter(
            window_size,
            stride,
            min_valid_fraction,
            required_curves=list(required_model_curves),
        )
        self.target_step_m = target_step_m

    def ingest(self, input_path: str | Path, out_dir: str | Path) -> IngestionResult:
        source = Path(input_path)
        files = sorted(source.rglob("*.las")) if source.is_dir() else [source]
        if not files:
            raise ValueError(f"no LAS files found under {source}")

        accepted_files = 0
        rejected_files = 0
        file_reports: list[dict[str, Any]] = []
        qc_reports: list[dict[str, object]] = []
        windows: list[CalibrationWindow] = []
        flat_frames: list[pd.DataFrame] = []
        for path in files:
            found_curves: list[str] = []
            try:
                las = lasio.read(path)
                well = self.canonicalizer.canonicalize(las, str(path))
                well = self.normalizer.normalize(well)
                useful = [
                    curve
                    for curve in well.curves.columns
                    if curve != "DEPT" and curve not in well.unusable_curves
                ]
                found_curves = useful
                missing = [curve for curve in self.required_model_curves if curve not in useful]
                if missing:
                    raise ValueError(f"missing required model channels: {', '.join(missing)}")
                well = self.resampler.resample(well)
                qc = self.qc.build(well.curves, well.well_id)
                well_windows = self.segmenter.segment(
                    well.curves,
                    qc,
                    well.well_id,
                    str(path),
                )
                if not well_windows:
                    raise ValueError(
                        "no windows passed the channel and valid-fraction requirements"
                    )
                accepted_files += 1
                windows.extend(well_windows)
                qc_reports.extend(qc.reports)
                flattened = well.curves.copy()
                flattened.insert(0, "well_id", well.well_id)
                flattened.insert(1, "source_file", str(path))
                flat_frames.append(flattened)
                file_reports.append(
                    {
                        "source_file": str(path),
                        "well_id": well.well_id,
                        "status": "accepted",
                        "curves": useful,
                        "original_mnemonics": well.original_mnemonics,
                        "canonical_units": well.units,
                        "window_count": len(well_windows),
                        "warnings": well.warnings,
                    }
                )
            except (OSError, ValueError, KeyError) as exc:
                rejected_files += 1
                file_reports.append(
                    {
                        "source_file": str(path),
                        "status": "rejected",
                        "reason": str(exc),
                        "found_curves": found_curves,
                    }
                )

        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        report = {
            "accepted_files": accepted_files,
            "rejected_files": rejected_files,
            "window_count": len(windows),
            "files": file_reports,
            "curves": qc_reports,
        }
        if not windows:
            (root / "ingestion_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            raise ValueError("ingestion produced no calibration windows")
        curves = pd.concat(flat_frames, ignore_index=True)
        write_parquet(curves, root / "curves.parquet")
        dataset = CalibrationDatasetBuilder().build(
            windows,
            root,
            dataset_id=f"calibration_{source.stem or source.name}",
            source=str(source),
            target_step_m=self.target_step_m,
            stride=self.segmenter.stride,
            qc_summary={"accepted_files": accepted_files, "rejected_files": rejected_files},
        )
        (root / "ingestion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        statistics = curve_statistics(curves, dataset.curve_names)
        (root / "statistics_report.json").write_text(
            json.dumps(statistics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return IngestionResult(dataset=dataset, report=report)
