from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.datasets import CurveAliasesConfig, IngestionPipeline
from synthetic_well_logs.datasets.reports import compare_synthetic_to_calibration
from synthetic_well_logs.generator import generate_well
from synthetic_well_logs.ml.train_autoencoder import train_autoencoder
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
    except (OSError, ValueError, RuntimeError, ValidationError) as exc:
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


@app.command(name="ingest-las")
def ingest_las_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True)],
    out: Annotated[Path, typer.Option("--out")],
    aliases: Annotated[
        Path,
        typer.Option("--aliases", exists=True, dir_okay=False),
    ] = Path("configs/curve_aliases.yaml"),
    target_step_m: Annotated[float, typer.Option("--target-step-m", min=0.001)] = 0.1,
    window_size: Annotated[int, typer.Option("--window-size", min=16)] = 128,
    stride: Annotated[int, typer.Option("--stride", min=1)] = 64,
) -> None:
    """Build a normalized Parquet/NPZ calibration dataset from LAS files."""
    try:
        pipeline = IngestionPipeline(
            CurveAliasesConfig.from_file(aliases),
            target_step_m=target_step_m,
            window_size=window_size,
            stride=stride,
        )
        result = pipeline.ingest(input_path, out)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"Calibration dataset: {result.dataset.manifest['well_count']} wells, "
        f"{result.dataset.manifest['window_count']} windows"
    )
    typer.echo(f"  manifest: {out / 'manifest.json'}")
    typer.echo(f"  metadata: {out / 'metadata.parquet'}")


@app.command(name="train-autoencoder")
def train_autoencoder_command(
    dataset: Annotated[Path, typer.Option("--dataset", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out")],
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 20,
    latent_dim: Annotated[int, typer.Option("--latent-dim", min=2)] = 32,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 64,
    learning_rate: Annotated[float, typer.Option("--learning-rate", min=1e-6)] = 0.001,
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    """Train and persist the Conv1D autoencoder realism artifact."""
    try:
        report = train_autoencoder(
            dataset,
            out,
            epochs=epochs,
            latent_dim=latent_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Training failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(
        f"Autoencoder trained: best epoch {report['best_epoch']}, "
        f"validation loss {report['val_loss'][report['best_epoch'] - 1]:.6f}"
    )
    typer.echo(f"  model: {out / 'model.pt'}")


@app.command(name="compare-stats")
def compare_stats_command(
    synthetic: Annotated[Path, typer.Option("--synthetic", exists=True, dir_okay=False)],
    calibration: Annotated[Path, typer.Option("--calibration", exists=True, file_okay=False)],
    out: Annotated[Path, typer.Option("--out")],
    strict: Annotated[bool, typer.Option("--strict")] = False,
) -> None:
    """Compare synthetic LAS statistics with a calibration dataset."""
    try:
        report = compare_synthetic_to_calibration(synthetic, calibration, out)
    except (OSError, ValueError, RuntimeError) as exc:
        typer.echo(f"Statistics comparison failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(f"Compared {len(report['deltas'])} curves: {report['gate']['status']}")
    typer.echo(f"  report: {out / 'validation_summary.md'}")
    if strict and report["gate"]["status"] == "failed":
        raise typer.Exit(code=1)
