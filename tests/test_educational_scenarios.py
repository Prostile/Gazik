from pathlib import Path

import pytest

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.validation import validate_export

ROOT = Path(__file__).parents[1]
SCENARIOS = [
    "coal_bed_detection.yaml",
    "shaly_sand_vs_siltstone.yaml",
    "carbonate_basic.yaml",
    "bad_hole_quality_control.yaml",
    "gas_sand_vs_tight_sand.yaml",
]


@pytest.mark.parametrize("name", SCENARIOS)
def test_educational_scenario_exports_and_validates(name: str, tmp_path: Path) -> None:
    scenario = ScenarioConfig.from_file(ROOT / "examples" / "educational" / name)
    paths = generate_well(scenario).export(tmp_path / Path(name).stem)
    assert set(paths) == {"las", "truth", "manifest", "preview"}
    assert all(path.exists() for path in paths.values())
    assert validate_export(paths["las"], paths["truth"])["valid"]

