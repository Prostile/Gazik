from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class DatasetFiles(DatasetModel):
    las: str
    report: str | None = None
    tops: str | None = None
    core: str | None = None
    interpreted: str | None = None


class DataQuality(DatasetModel):
    trusted: bool
    comments: list[str] = Field(default_factory=list)


class LicenseInfo(DatasetModel):
    usage: str
    redistribution_allowed: bool


class WellMetadata(DatasetModel):
    well_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    country: str | None = None
    basin: str | None = None
    field: str | None = None
    files: DatasetFiles
    curve_mnemonics: dict[str, str] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)
    data_quality: DataQuality
    license: LicenseInfo


def load_well_metadata(path: str | Path) -> WellMetadata:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: metadata root must be an object")
    return WellMetadata.model_validate(payload)


def _validate_csv(path: Path, required: set[str], label: str) -> list[str]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        return [f"{path}: cannot read {label} CSV: {exc}"]
    missing = required - set(frame.columns)
    errors = [f"{path}: missing {label} columns {sorted(missing)}"] if missing else []
    if not missing:
        invalid_depth = (frame["base"] <= frame["top"]) | frame[["top", "base"]].isna().any(
            axis=1
        )
        if invalid_depth.any():
            errors.append(f"{path}: every {label} interval must have finite top < base")
    return errors


def validate_dataset_structure(root: str | Path) -> dict[str, Any]:
    directory = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    metadata_paths = sorted(directory.rglob("well_metadata.yaml")) if directory.exists() else []
    if not directory.exists():
        errors.append(f"dataset root does not exist: {directory}")
    elif not metadata_paths:
        errors.append(f"no well_metadata.yaml files found under {directory}")
    wells: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        try:
            metadata = load_well_metadata(metadata_path)
        except (OSError, ValueError, ValidationError) as exc:
            errors.append(f"{metadata_path}: invalid metadata: {exc}")
            continue
        well_errors: list[str] = []
        resolved: dict[str, str] = {}
        for name, reference in metadata.files.model_dump().items():
            if not reference:
                continue
            target = (metadata_path.parent / reference).resolve()
            resolved[name] = str(target)
            if not target.exists():
                message = f"{metadata_path}: referenced {name} file does not exist: {target}"
                well_errors.append(message)
        las_path = Path(resolved["las"])
        if las_path.exists() and las_path.suffix.lower() != ".las":
            well_errors.append(f"{las_path}: LAS reference must use the .las extension")
        tops_path = resolved.get("tops")
        if tops_path and Path(tops_path).exists():
            well_errors.extend(
                _validate_csv(
                    Path(tops_path),
                    {"well_id", "top", "base", "name", "type", "confidence"},
                    "tops",
                )
            )
        interpreted_paths: list[Path] = []
        if resolved.get("interpreted") and Path(resolved["interpreted"]).exists():
            interpreted_paths.append(Path(resolved["interpreted"]))
        interpreted_paths.extend(
            path
            for path in metadata_path.parent.parent.glob("interpreted/*_lithology.csv")
            if path not in interpreted_paths
        )
        for interpreted_path in interpreted_paths:
            well_errors.extend(
                _validate_csv(
                    interpreted_path,
                    {"well_id", "top", "base", "lithology", "facies", "source", "confidence"},
                    "lithology",
                )
            )
        errors.extend(well_errors)
        wells.append(
            {
                "well_id": metadata.well_id,
                "metadata": str(metadata_path),
                "valid": not well_errors,
                "files": resolved,
            }
        )
    return {
        "valid": not errors,
        "root": str(directory),
        "well_count": len(wells),
        "wells": wells,
        "errors": errors,
        "warnings": warnings,
    }
