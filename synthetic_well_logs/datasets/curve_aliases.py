from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import pandas as pd
import yaml

DEFAULT_ALIAS_PATH = Path(__file__).parents[2] / "configs" / "curve_aliases.yaml"


@dataclass(frozen=True, slots=True)
class CurveAlias:
    canonical: str
    aliases: tuple[str, ...]
    unit: str | None = None
    required: bool = False


@dataclass(slots=True)
class CurveAliasesConfig:
    curves: dict[str, CurveAlias]

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_ALIAS_PATH) -> CurveAliasesConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        curves = {
            canonical.upper(): CurveAlias(
                canonical=canonical.upper(),
                aliases=tuple(str(item).upper() for item in config["aliases"]),
                unit=config.get("unit"),
                required=bool(config.get("required", False)),
            )
            for canonical, config in payload.items()
        }
        return cls(curves=curves)

    def canonical_name(self, mnemonic: str) -> str | None:
        normalized = mnemonic.strip().upper()
        for canonical, entry in self.curves.items():
            if normalized in entry.aliases:
                return canonical
        return None


@dataclass(slots=True)
class CanonicalWell:
    well_id: str
    source_file: str
    curves: pd.DataFrame
    units: dict[str, str]
    original_mnemonics: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    unusable_curves: list[str] = field(default_factory=list)


class CurveCanonicalizer:
    def __init__(self, aliases_config: CurveAliasesConfig):
        self.aliases = aliases_config

    def canonicalize(self, las: lasio.LASFile, source_file: str = "") -> CanonicalWell:
        selected: dict[str, np.ndarray] = {}
        units: dict[str, str] = {}
        originals: dict[str, str] = {}
        warnings: list[str] = []

        for curve in las.curves:
            canonical = self.aliases.canonical_name(curve.mnemonic)
            if canonical is None or canonical in selected:
                continue
            selected[canonical] = np.asarray(las[curve.mnemonic], dtype=float)
            units[canonical] = str(curve.unit or "").strip()
            originals[canonical] = curve.mnemonic

        if "DEPT" not in selected:
            index_mnemonic = las.curves[0].mnemonic if las.curves else ""
            if self.aliases.canonical_name(index_mnemonic) == "DEPT":
                selected["DEPT"] = np.asarray(las.index, dtype=float)
                units["DEPT"] = str(las.curves[0].unit or "").strip()
                originals["DEPT"] = index_mnemonic
            else:
                raise ValueError("LAS does not contain a DEPT/DEPTH/MD curve")

        useful = sorted(set(selected) - {"DEPT"})
        missing = sorted(set(self.aliases.curves) - set(selected))
        if len(useful) < 3:
            warnings.append(f"only {len(useful)} useful canonical curves found")
        if missing:
            warnings.append(f"missing canonical curves: {', '.join(missing)}")

        well_item = getattr(las.well, "WELL", None)
        field_item = getattr(las.well, "FLD", None)
        country_item = getattr(las.well, "CTRY", None)
        well_name = str(well_item.value if well_item is not None else "").strip()
        well_id = well_name or Path(source_file).stem or "UNKNOWN_WELL"
        metadata = {
            "field": str(field_item.value if field_item is not None else "").strip(),
            "country": str(country_item.value if country_item is not None else "").strip(),
        }
        return CanonicalWell(
            well_id=well_id,
            source_file=source_file,
            curves=pd.DataFrame(selected),
            units=units,
            original_mnemonics=originals,
            metadata=metadata,
            warnings=warnings,
        )
