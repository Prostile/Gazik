from pathlib import Path

import numpy as np

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.config import ResistivityModelConfig
from synthetic_well_logs.forward.model import calculate_resistivity

ROOT = Path(__file__).parents[1]


def _rt(vsh: list[float]) -> np.ndarray:
    size = len(vsh)
    return calculate_resistivity(
        phi=np.full(size, 0.18),
        sw=np.full(size, 0.45),
        vsh=np.asarray(vsh),
        lithology=np.full(size, "sandstone"),
        fluid=np.full(size, "oil"),
        config=ResistivityModelConfig(model="shaly_sand", shale_conductivity_factor=0.5),
    )


def test_shaly_sand_rt_lower_than_clean_sand_for_same_phi_sw() -> None:
    clean, shaly = _rt([0.1, 0.5])
    assert clean > shaly


def test_high_vsh_reduces_effective_resistivity() -> None:
    values = _rt([0.1, 0.4, 0.8])
    assert np.all(np.diff(values) < 0)


def test_siltstone_and_coal_are_not_marked_pay() -> None:
    scenario = ScenarioConfig.from_file(ROOT / "examples/educational/coal_bed_detection.yaml")
    well = generate_well(scenario)
    assert not well.truth.is_pay[well.truth.facies == "siltstone"].any()
    assert not well.truth.is_pay[well.truth.facies == "coal"].any()

