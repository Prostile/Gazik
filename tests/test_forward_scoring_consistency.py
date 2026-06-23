import numpy as np

from synthetic_well_logs.config import ResistivityModelConfig
from synthetic_well_logs.constraints.scoring import (
    TruthSlice,
    expected_curves_for_slice,
    score_curves_against_truth,
)
from synthetic_well_logs.forward.model import calculate_expected_curves_from_arrays


def test_scoring_uses_same_shaly_sand_resistivity_as_forward_model() -> None:
    config = ResistivityModelConfig(model="shaly_sand", shale_conductivity_factor=0.5)
    size = 12
    truth = TruthSlice(
        vsh=np.full(size, 0.75),
        phi=np.full(size, 0.16),
        sw=np.full(size, 0.45),
        facies=np.full(size, "shaly_sandstone"),
        lithology=np.full(size, "sandstone"),
        fluid=np.full(size, "oil"),
        is_pay=np.full(size, True),
        is_reservoir=np.full(size, True),
    )
    forward = calculate_expected_curves_from_arrays(
        vsh=truth.vsh,
        phi=truth.phi,
        sw=truth.sw,
        lithology=truth.lithology,
        fluid=truth.fluid,
        resistivity_config=config,
    )
    scoring_expected = expected_curves_for_slice(truth, resistivity_config=config)
    assert np.allclose(scoring_expected["RT"], forward["RT"])
    assert not np.allclose(scoring_expected["RT"], expected_curves_for_slice(truth)["RT"])
    score = score_curves_against_truth(
        forward,
        truth,
        resistivity_config=config,
        include_facies_ranges=False,
    )
    assert score.components["rt_sw_violation"] == 0.0

