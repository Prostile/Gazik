import json
from pathlib import Path

import lasio

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.validation import validate_export

ROOT = Path(__file__).parents[1]


def test_export_roundtrip(tmp_path: Path) -> None:
    scenario = ScenarioConfig.from_file(ROOT / "examples/example_scenario_gas_sand.yaml")
    paths = generate_well(scenario).export(tmp_path / "well_001")

    assert all(path.exists() for path in paths.values())
    las = lasio.read(paths["las"])
    assert list(las.keys()) == scenario.curves
    assert len(las["DEPT"]) == 3001

    truth = json.loads(paths["truth"].read_text(encoding="utf-8"))
    assert truth["well_id"] == scenario.well.name
    assert len(truth["samples"]["depth"]) == 3001
    assert any(interval["is_pay"] for interval in truth["intervals"])

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["generator_version"] == "0.2.0"
    assert set(manifest["sha256"]) == {"las", "truth", "preview"}

    report = validate_export(paths["las"], paths["truth"])
    assert report["valid"], report["errors"]
