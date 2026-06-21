from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pandas as pd

from synthetic_well_logs.datasets.calibration_dataset import CalibrationDataset
from synthetic_well_logs.datasets.qc import QC_RANGES


def curve_statistics(frame: pd.DataFrame, curves: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for curve in curves:
        values = frame[curve].to_numpy(dtype=float) if curve in frame else np.array([])
        finite = values[np.isfinite(values)]
        if not finite.size:
            stats[curve] = {"count": 0, "missing_rate": 1.0}
            continue
        low, high = QC_RANGES.get(curve, (-np.inf, np.inf))
        stats[curve] = {
            "count": int(finite.size),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "p05": float(np.quantile(finite, 0.05)),
            "p50": float(np.quantile(finite, 0.50)),
            "p95": float(np.quantile(finite, 0.95)),
            "missing_rate": float(1 - finite.size / max(values.size, 1)),
            "range_violation_rate": float(np.mean((finite < low) | (finite > high))),
            "autocorrelation_lag1": _autocorrelation(finite),
        }
    available = [curve for curve in curves if curve in frame]
    correlation = frame[available].corr(min_periods=2).fillna(0.0)
    crossplots = {
        f"{left}_vs_{right}": _crossplot_statistics(frame, left, right)
        for left, right in (("RHOB", "NPHI"), ("GR", "RT"), ("RHOB", "DT"))
        if left in frame and right in frame
    }
    return {
        "curves": stats,
        "correlation_matrix": correlation.to_dict(),
        "crossplots": crossplots,
    }


def _autocorrelation(values: np.ndarray) -> float:
    if values.size < 3 or np.std(values) == 0:
        return 0.0
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _crossplot_statistics(frame: pd.DataFrame, left: str, right: str) -> dict[str, float | int]:
    pairs = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pairs) < 2:
        return {"count": int(len(pairs)), "correlation": 0.0, "slope": 0.0, "intercept": 0.0}
    x = pairs[left].to_numpy(dtype=float)
    y = pairs[right].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    correlation = 0.0 if np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])
    return {
        "count": int(len(pairs)),
        "correlation": correlation,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def calibration_frame(dataset: CalibrationDataset) -> pd.DataFrame:
    arrays, _ = dataset.load_arrays()
    raw = dataset.denormalize(arrays)
    return pd.DataFrame(
        {curve: raw[:, channel, :].reshape(-1) for channel, curve in enumerate(dataset.curve_names)}
    )


def compare_synthetic_to_calibration(
    synthetic_path: str | Path,
    calibration_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    dataset = CalibrationDataset.open(calibration_path)
    real_frame = calibration_frame(dataset)
    las = lasio.read(synthetic_path)
    las_curves = set(las.keys())
    synthetic_frame = pd.DataFrame(
        {
            curve: np.asarray(las[curve], dtype=float)
            for curve in dataset.curve_names
            if curve in las_curves
        }
    )
    real_stats = curve_statistics(real_frame, dataset.curve_names)
    synthetic_stats = curve_statistics(synthetic_frame, dataset.curve_names)
    deltas: dict[str, dict[str, float]] = {}
    for curve in dataset.curve_names:
        if real_stats["curves"][curve].get("count", 0) and synthetic_stats["curves"][curve].get(
            "count", 0
        ):
            deltas[curve] = {
                metric: float(
                    synthetic_stats["curves"][curve][metric] - real_stats["curves"][curve][metric]
                )
                for metric in ("mean", "std", "p05", "p50", "p95")
            }

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "calibration_statistics.json": real_stats,
        "synthetic_statistics.json": synthetic_stats,
        "synthetic_vs_real_statistics.json": {"deltas": deltas},
        "correlation_matrix_real.json": real_stats["correlation_matrix"],
        "correlation_matrix_synthetic.json": synthetic_stats["correlation_matrix"],
    }
    for filename, payload in payloads.items():
        (root / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    lines = [
        "# Synthetic vs calibration validation",
        "",
        "| Curve | Δ mean | Δ std |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {curve} | {metrics['mean']:.5g} | {metrics['std']:.5g} |"
        for curve, metrics in deltas.items()
    )
    (root / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"real": real_stats, "synthetic": synthetic_stats, "deltas": deltas}
