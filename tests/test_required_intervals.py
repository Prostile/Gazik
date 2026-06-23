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


def test_required_interval_provenance_is_written_to_truth() -> None:
    scenario = ScenarioConfig.from_file(ROOT / "examples/educational/coal_bed_detection.yaml")
    well = generate_well(scenario)
    injected = [item for item in well.truth.intervals if item.get("injection_source") == "required_interval"]
    assert injected
    roles = {item["facies"]: item["injected_role"] for item in injected}
    assert roles["coal"] == "distractor"
    assert roles["siltstone"] == "comparison"
    assert all(item["required_interval_index"] is not None for item in injected)
    assert all(item["placement"] in {"deterministic", "random"} for item in injected)


def test_random_required_interval_placement_is_supported() -> None:
    config = ScenarioConfig.model_validate(
        {
            "well": {"name": "RANDOM_REQUIRED_PLACEMENT"},
            "depth": {"start": 0.0, "stop": 100.0, "step": 0.5, "unit": "m"},
            "geology": {
                "depositional_environment": "test",
                "stacking_pattern": "alternation",
                "facies_set": ["shale", "clean_sandstone", "siltstone"],
            },
            "target": {"reservoir_type": "clean_sandstone", "hydrocarbon": "oil"},
            "curves": ["DEPT", "GR", "RHOB", "NPHI", "DT", "RT"],
            "required_intervals": [
                {
                    "facies": "siltstone",
                    "thickness_m": [5.0, 5.0],
                    "count": 1,
                    "role": "comparison",
                    "placement": "random",
                }
            ],
            "realism": {"mode": "none"},
            "seed": 123,
        }
    )
    well = generate_well(config)
    injected = [item for item in well.truth.intervals if item.get("injection_source") == "required_interval"]
    assert len(injected) == 1
    assert injected[0]["placement"] == "random"
    assert injected[0]["injected_role"] == "comparison"


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
