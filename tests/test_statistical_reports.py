from pathlib import Path

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.datasets.reports import compare_synthetic_to_calibration


def test_synthetic_vs_calibration_report(tmp_path: Path, calibration_dataset: Path) -> None:
    scenario = ScenarioConfig.from_file("examples/example_scenario_gas_sand.yaml")
    paths = generate_well(scenario).export(tmp_path / "synthetic")
    report_dir = tmp_path / "reports"
    report = compare_synthetic_to_calibration(paths["las"], calibration_dataset, report_dir)
    assert set(report["deltas"]) == {"GR", "RHOB", "NPHI", "DT", "RT"}
    assert (report_dir / "validation_summary.md").exists()
    assert "RHOB_vs_NPHI" in report["real"]["crossplots"]
    assert "GR_vs_RT" in report["synthetic"]["crossplots"]
