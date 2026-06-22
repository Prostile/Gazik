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

