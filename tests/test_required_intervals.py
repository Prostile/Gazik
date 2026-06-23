from pathlib import Path

import pytest

from synthetic_well_logs import ScenarioConfig, generate_well

ROOT = Path(__file__).parents[1]


def test_required_facies_are_injected() -> None:
    expected_by_scenario = {
        "coal_bed_detection.yaml": {"coal", "siltstone"},
        "shaly_sand_vs_siltstone.yaml": {"siltstone"},
        "carbonate_basic.yaml": {"marl", "anhydrite"},
        "bad_hole_quality_control.yaml": {"siltstone"},
        "gas_sand_vs_tight_sand.yaml": {"tight_sandstone"},
    }
    for scenario_name, required_facies in expected_by_scenario.items():
        scenario = ScenarioConfig.from_file(ROOT / "examples" / "educational" / scenario_name)
        well = generate_well(scenario)
        assert required_facies <= set(well.truth.facies)


def test_required_intervals_do_not_overwrite_target() -> None:
    scenario = ScenarioConfig.from_file(ROOT / "examples/educational/gas_sand_vs_tight_sand.yaml")
    well = generate_well(scenario)
    hydrocarbon_mask = well.truth.fluid == scenario.target.hydrocarbon
    assert hydrocarbon_mask.any()
    assert set(well.truth.facies[hydrocarbon_mask]) == {scenario.target.reservoir_type}
    assert "tight_sandstone" in set(well.truth.facies[~hydrocarbon_mask])


def test_generation_fails_when_required_intervals_do_not_fit() -> None:
    config = ScenarioConfig.model_validate(
        {
            "well": {"name": "TOO_SMALL"},
            "depth": {"start": 0.0, "stop": 10.0, "step": 1.0, "unit": "m"},
            "geology": {
                "depositional_environment": "test",
                "stacking_pattern": "alternation",
                "facies_set": ["shale", "clean_sandstone"],
            },
            "target": {"reservoir_type": "clean_sandstone", "hydrocarbon": "oil"},
            "curves": ["DEPT", "GR", "RHOB", "NPHI", "DT", "RT"],
            "required_intervals": [
                {"facies": "shale", "thickness_m": [20.0, 20.0], "count": 1}
            ],
            "realism": {"mode": "none"},
        }
    )
    with pytest.raises(ValueError, match="cannot fit required interval"):
        generate_well(config)
