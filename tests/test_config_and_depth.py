from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from synthetic_well_logs import ScenarioConfig
from synthetic_well_logs.generator import create_depth_grid

ROOT = Path(__file__).parents[1]


def test_yaml_scenario_and_depth_grid() -> None:
    scenario = ScenarioConfig.from_file(ROOT / "examples/example_scenario_gas_sand.yaml")
    depth = create_depth_grid(scenario)
    assert len(depth) == 3001
    assert depth[0] == 900.0
    assert depth[-1] == 1200.0
    assert np.allclose(np.diff(depth), 0.1)


def test_invalid_depth_step_is_rejected() -> None:
    payload = ScenarioConfig.from_file(
        ROOT / "examples/example_scenario_gas_sand.yaml"
    ).model_dump()
    payload["depth"]["step"] = 0
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(payload)


def test_target_must_be_in_facies_set() -> None:
    payload = ScenarioConfig.from_file(
        ROOT / "examples/example_scenario_gas_sand.yaml"
    ).model_dump()
    payload["target"]["reservoir_type"] = "dolomite"
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(payload)

