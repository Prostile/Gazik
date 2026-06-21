from __future__ import annotations

import numpy as np

from synthetic_well_logs.datasets.curve_aliases import CanonicalWell


def _unit_key(unit: str) -> str:
    return unit.lower().replace(" ", "").replace("³", "3").replace(".", "")


class UnitNormalizer:
    CANONICAL_UNITS = {
        "DEPT": "m",
        "GR": "API",
        "RHOB": "g/cc",
        "NPHI": "v/v",
        "DT": "us/ft",
        "RT": "ohm.m",
    }

    def normalize(self, well: CanonicalWell) -> CanonicalWell:
        result = well.curves.copy(deep=True)
        warnings = list(well.warnings)
        unusable = list(well.unusable_curves)
        for curve in result.columns:
            values = result[curve].to_numpy(dtype=float)
            unit = _unit_key(well.units.get(curve, ""))
            converted, recognized = self._convert(curve, values, unit)
            if not recognized:
                converted, safe = self._heuristic(curve, values)
                if safe:
                    warnings.append(
                        f"{curve}: unknown unit '{well.units.get(curve, '')}', safe heuristic used"
                    )
                else:
                    warnings.append(
                        f"{curve}: unknown unit '{well.units.get(curve, '')}', curve unusable"
                    )
                    unusable.append(curve)
                    converted = np.full_like(values, np.nan)
            result[curve] = converted

        well.curves = result
        well.units = {curve: self.CANONICAL_UNITS[curve] for curve in result.columns}
        well.warnings = warnings
        well.unusable_curves = sorted(set(unusable))
        return well

    @staticmethod
    def _convert(curve: str, values: np.ndarray, unit: str) -> tuple[np.ndarray, bool]:
        if curve == "DEPT":
            if unit in {"m", "meter", "meters", "metre", "metres"}:
                return values, True
            if unit in {"ft", "feet", "foot"}:
                return values * 0.3048, True
        elif curve == "NPHI":
            if unit in {"v/v", "vv", "dec", "fraction", ""}:
                return values, True
            if unit in {"%", "pct", "percent", "pu"}:
                return values / 100.0, True
        elif curve == "RHOB":
            if unit in {"g/cc", "g/cm3", "g/c3", "gcc", "gm/cc"}:
                return values, True
            if unit in {"kg/m3", "kgm3"}:
                return values / 1000.0, True
        elif curve == "DT":
            if unit in {"us/ft", "us/f", "µs/ft", "usec/ft"}:
                return values, True
            if unit in {"us/m", "µs/m", "usec/m"}:
                return values * 0.3048, True
        elif curve == "RT":
            if unit in {"ohmm", "ohm-m", "ohm·m", "ohm/m", ""}:
                return values, True
        elif curve == "GR" and unit in {"api", "gapi", ""}:
            return values, True
        return values, False

    @staticmethod
    def _heuristic(curve: str, values: np.ndarray) -> tuple[np.ndarray, bool]:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return values, False
        median = float(np.median(np.abs(finite)))
        if curve == "NPHI" and 1.5 < median <= 100:
            return values / 100.0, True
        if curve == "RHOB" and 1000 <= median <= 3500:
            return values / 1000.0, True
        if curve == "DEPT" and median >= 0:
            return values, True
        if curve == "GR" and median <= 300:
            return values, True
        if curve == "RT" and np.nanmin(finite) >= 0:
            return values, True
        if curve == "DT" and 30 <= median <= 250:
            return values, True
        if curve == "RHOB" and 1 <= median <= 4:
            return values, True
        if curve == "NPHI" and median <= 1:
            return values, True
        return values, False
