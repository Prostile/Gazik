from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lasio
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from synthetic_well_logs import __version__
from synthetic_well_logs.domain import GeneratedWell, GroundTruth
from synthetic_well_logs.rocks import FACIES_DISPLAY_NAMES_RU

CURVE_META = {
    "DEPT": ("m", "Measured depth"),
    "GR": ("API", "Gamma ray"),
    "CALI": ("in", "Caliper"),
    "RHOB": ("g/cc", "Bulk density"),
    "NPHI": ("v/v", "Neutron porosity"),
    "DT": ("us/ft", "Compressional slowness"),
    "RT": ("ohm.m", "Deep resistivity"),
}


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _float_list(values: np.ndarray) -> list[float]:
    return np.round(values.astype(float), 6).tolist()


def _truth_payload(truth: GroundTruth) -> dict[str, Any]:
    return {
        "well_id": truth.well_id,
        "depth_unit": truth.depth_unit,
        "intervals": truth.intervals,
        "contacts": truth.contacts,
        "artifacts": truth.artifacts,
        "expected_student_outputs": {
            "tops": True,
            "lithology": True,
            "vsh": True,
            "porosity": True,
            "sw": True,
            "net_pay": True,
            "bad_hole": True,
        },
        "samples": {
            "depth": _float_list(truth.depth),
            "facies": truth.facies.tolist(),
            "facies_display_name_ru": [FACIES_DISPLAY_NAMES_RU[item] for item in truth.facies],
            "lithology": truth.lithology.tolist(),
            "vsh": _float_list(truth.vsh),
            "phi": _float_list(truth.phi),
            "sw": _float_list(truth.sw),
            "fluid": truth.fluid.tolist(),
            "is_reservoir": truth.is_reservoir.astype(bool).tolist(),
            "is_pay": truth.is_pay.astype(bool).tolist(),
            "bad_hole_mask": truth.bad_hole_mask.astype(bool).tolist(),
        },
    }


def _write_las(well: GeneratedWell, path: Path) -> None:
    las = lasio.LASFile()
    las.version.VERS = 2.0
    las.version.WRAP = "NO"
    las.well.WELL = well.scenario.well.name
    las.well.FLD = well.scenario.well.field
    las.well.CTRY = well.scenario.well.country
    las.well.STRT = float(well.depth[0])
    las.well.STOP = float(well.depth[-1])
    las.well.STEP = well.scenario.depth.step
    las.well.NULL = -999.25
    for curve in well.curves.columns:
        unit, description = CURVE_META[curve]
        if curve == "DEPT":
            unit = well.scenario.depth.unit
        las.append_curve(curve, well.curves[curve].to_numpy(), unit=unit, descr=description)
    las.write(str(path), version=2.0, fmt="%.6f")


def _write_preview(well: GeneratedWell, path: Path) -> None:
    visible_curves = [curve for curve in well.curves.columns if curve != "DEPT"]
    figure = make_subplots(
        rows=1,
        cols=len(visible_curves),
        shared_yaxes=True,
        horizontal_spacing=0.018,
        subplot_titles=visible_curves,
    )
    colors = {
        "GR": "#3a8b63",
        "CALI": "#bc5b46",
        "RHOB": "#315f8f",
        "NPHI": "#8961a7",
        "DT": "#b17c24",
        "RT": "#252b33",
    }
    for column, curve in enumerate(visible_curves, start=1):
        figure.add_trace(
            go.Scattergl(
                x=well.curves[curve],
                y=well.depth,
                mode="lines",
                line={"color": colors[curve], "width": 1.1},
                name=curve,
                showlegend=False,
                connectgaps=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(showgrid=True, gridcolor="#e8ecef", row=1, col=column)
        if curve == "RT":
            figure.update_xaxes(type="log", dtick=1, row=1, col=column)
        figure.update_xaxes(title_text=CURVE_META[curve][0], row=1, col=column)
    figure.update_yaxes(autorange="reversed")
    figure.update_yaxes(
        title_text=f"Depth ({well.scenario.depth.unit})",
        row=1,
        col=1,
    )
    figure.update_layout(
        title={
            "text": (
                f"{well.well_id}<br><sup>Породы сценария: "
                + ", ".join(
                    FACIES_DISPLAY_NAMES_RU[item] for item in well.scenario.geology.facies_set
                )
                + "</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        height=900,
        width=max(1100, len(visible_curves) * 190),
        margin={"l": 70, "r": 30, "t": 85, "b": 45},
    )
    html = figure.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    document_title = f"Well preview — {well.well_id}"
    html = html.replace(
        "<head>",
        f'<head><meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{document_title}</title>",
        1,
    )
    path.write_text(html, encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_generated_well(well: GeneratedWell, prefix: Path) -> dict[str, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "las": prefix.with_suffix(".las"),
        "truth": prefix.parent / f"{prefix.name}_truth.json",
        "manifest": prefix.parent / f"{prefix.name}_manifest.json",
        "preview": prefix.parent / f"{prefix.name}_preview.html",
    }
    _write_las(well, paths["las"])
    _json_dump(paths["truth"], _truth_payload(well.truth))
    _write_preview(well, paths["preview"])
    manifest = {
        "well_id": well.well_id,
        "scenario_id": prefix.name,
        "generator_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": well.scenario.seed,
        "depth": well.scenario.depth.model_dump(),
        "curves": list(well.curves.columns),
        "required_intervals": [
            {
                "facies": item.facies,
                "count": item.count,
                "role": item.role,
                "placement": item.placement,
                "thickness_m": list(item.thickness_m),
            }
            for item in well.scenario.required_intervals
        ],
        "facies": [
            {"id": item, "display_name_ru": FACIES_DISPLAY_NAMES_RU[item]}
            for item in well.scenario.geology.facies_set
        ],
        "files": {name: path.name for name, path in paths.items() if name != "manifest"},
        "sha256": {name: _sha256(path) for name, path in paths.items() if name != "manifest"},
        "validation": well.validation,
    }
    _json_dump(paths["manifest"], manifest)
    return paths
