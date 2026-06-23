from pathlib import Path

import numpy as np
import pytest

from synthetic_well_logs import ScenarioConfig, generate_well

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def coal_well():
    scenario = ScenarioConfig.from_file(ROOT / "examples/educational/coal_bed_detection.yaml")
    return generate_well(scenario)


@pytest.fixture(scope="module")
def carbonate_well():
    scenario = ScenarioConfig.from_file(ROOT / "examples/educational/carbonate_basic.yaml")
    return generate_well(scenario)


def _median(well, curve: str, facies: str) -> float:
    mask = well.truth.facies == facies
    assert np.any(mask), facies
    return float(well.curves.loc[mask, curve].median())


def test_coal_response_and_pay_rule(coal_well) -> None:
    assert _median(coal_well, "RHOB", "coal") < _median(coal_well, "RHOB", "clean_sandstone")
    assert _median(coal_well, "RT", "coal") > 5.0
    coal = coal_well.truth.facies == "coal"
    assert not coal_well.truth.is_pay[coal].any()


def test_siltstone_gr_exceeds_clean_sand(coal_well) -> None:
    assert _median(coal_well, "GR", "siltstone") > _median(
        coal_well, "GR", "clean_sandstone"
    )


def test_carbonate_special_responses(carbonate_well) -> None:
    assert _median(carbonate_well, "RHOB", "anhydrite") > _median(
        carbonate_well, "RHOB", "limestone"
    )
    assert _median(carbonate_well, "GR", "marl") > _median(carbonate_well, "GR", "limestone")
    anhydrite = carbonate_well.truth.facies == "anhydrite"
    assert not carbonate_well.truth.is_reservoir[anhydrite].any()
    marl = carbonate_well.truth.facies == "marl"
    assert not carbonate_well.truth.is_reservoir[marl].any()


def test_marl_explicit_target_can_be_reservoir() -> None:
    scenario = ScenarioConfig.model_validate(
        {
            "well": {"name": "MARL_TARGET"},
            "depth": {"start": 0.0, "stop": 80.0, "step": 0.2, "unit": "m"},
            "geology": {
                "depositional_environment": "carbonate",
                "stacking_pattern": "alternation",
                "facies_set": ["shale", "marl"],
            },
            "target": {"reservoir_type": "marl", "hydrocarbon": "oil"},
            "curves": ["DEPT", "GR", "RHOB", "NPHI", "DT", "RT"],
            "realism": {"mode": "none"},
        }
    )
    well = generate_well(scenario)
    target = well.truth.facies == "marl"
    assert well.truth.is_reservoir[target].any()


def test_siltstone_can_be_reservoir_with_good_phi_and_vsh() -> None:
    scenario = ScenarioConfig.model_validate(
        {
            "well": {"name": "SILTSTONE_TARGET"},
            "depth": {"start": 0.0, "stop": 80.0, "step": 0.2, "unit": "m"},
            "geology": {
                "depositional_environment": "clastic",
                "stacking_pattern": "alternation",
                "facies_set": ["shale", "siltstone"],
            },
            "target": {
                "reservoir_type": "siltstone",
                "hydrocarbon": "oil",
                "porosity_range": [0.12, 0.18],
                "water_saturation_range": [0.35, 0.55],
            },
            "curves": ["DEPT", "GR", "RHOB", "NPHI", "DT", "RT"],
            "realism": {"mode": "none"},
        }
    )
    well = generate_well(scenario)
    target = (well.truth.facies == "siltstone") & (well.truth.phi >= 0.12) & (well.truth.vsh < 0.45)
    assert target.any()
    assert well.truth.is_reservoir[target].all()
