from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lasio
import numpy as np

from synthetic_well_logs.constraints.physics_constraints import RANGES


def validate_export(well_path: str | Path, truth_path: str | Path) -> dict[str, Any]:
    las = lasio.read(well_path)
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    curve_names = list(las.keys())
    errors: list[str] = []
    warnings: list[str] = []
    depth = np.asarray(las["DEPT"], dtype=float) if "DEPT" in curve_names else np.array([])
    if not depth.size:
        errors.append("LAS does not contain DEPT")
    elif depth.size < 2 or not np.all(np.diff(depth) > 0):
        errors.append("depth is not strictly monotonic")
    for curve in curve_names:
        if len(las[curve]) != len(depth):
            errors.append(f"curve {curve} length does not match DEPT")
        if curve in RANGES:
            finite = np.asarray(las[curve], dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size and (finite.min() < RANGES[curve][0] or finite.max() > RANGES[curve][1]):
                warnings.append(f"curve {curve} contains values outside QC range")
    intervals = truth.get("intervals", [])
    if not intervals:
        errors.append("truth contains no intervals")
    else:
        for left, right in zip(intervals, intervals[1:], strict=False):
            if not np.isclose(left["base"], right["top"], atol=1e-6):
                errors.append("truth intervals contain a gap or overlap")
                break
    for name, values in truth.get("samples", {}).items():
        if len(values) != len(depth):
            errors.append(f"truth sample array {name} length does not match DEPT")
    return {
        "valid": not errors,
        "categories": {
            "structural": {"valid": not errors, "errors": errors},
            "physical": {"valid": not warnings, "warnings": warnings},
            "export": {"valid": "DEPT" in curve_names and bool(depth.size)},
            "educational": {"valid": bool(intervals)},
        },
        "errors": errors,
        "warnings": warnings,
        "sample_count": int(len(depth)),
        "curves": curve_names,
        "artifact_count": len(truth.get("artifacts", [])),
    }
