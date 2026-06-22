from pathlib import Path

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.constraints.reports import FACIES_CURVE_RANGES
from synthetic_well_logs.forward.model import MATRIX_DENSITY, MATRIX_DT
from synthetic_well_logs.petrophysics.truth_generator import LITHOLOGY, PRIORS

ROOT = Path(__file__).parents[1]
NEW_ROCKS = {"siltstone", "marl", "coal", "anhydrite"}


def test_new_rocks_have_complete_properties() -> None:
    assert PRIORS.keys() >= NEW_ROCKS
    assert LITHOLOGY.keys() >= NEW_ROCKS
    assert MATRIX_DENSITY.keys() >= NEW_ROCKS
    assert MATRIX_DT.keys() >= NEW_ROCKS
    assert FACIES_CURVE_RANGES.keys() >= NEW_ROCKS


def test_all_new_rocks_generate() -> None:
    scenario = ScenarioConfig.from_file(ROOT / "examples/educational/carbonate_basic.yaml")
    well = generate_well(scenario)
    assert {"marl", "anhydrite"} <= set(well.truth.facies)
