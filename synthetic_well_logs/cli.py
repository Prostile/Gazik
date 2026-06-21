from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.generator import generate_well
from synthetic_well_logs.validation import validate_export

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Controlled synthetic well-log generator.",
)


@app.command()
def generate(
    scenario: Annotated[Path, typer.Option("--scenario", "-s", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output file prefix")],
) -> None:
    """Generate LAS, hidden truth, manifest and interactive preview."""
    try:
        config = ScenarioConfig.from_file(scenario)
        well = generate_well(config)
        paths = well.export(out)
    except (OSError, ValueError, ValidationError) as exc:
        typer.echo(f"Generation failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Generated {well.well_id} ({len(well.depth)} depth samples)")
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")


@app.command(name="validate")
def validate_command(
    well: Annotated[Path, typer.Option("--well", exists=True, dir_okay=False)],
    truth: Annotated[Path, typer.Option("--truth", exists=True, dir_okay=False)],
) -> None:
    """Validate an exported LAS file against its hidden truth JSON."""
    report = validate_export(well, truth)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise typer.Exit(code=1)
