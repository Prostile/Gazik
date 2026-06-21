from copy import deepcopy

import numpy as np
import torch

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.ml.autoencoder import AutoencoderConfig, Conv1dAutoencoder
from synthetic_well_logs.realism import AutoencoderMCMCRealismEnhancer


def test_autoencoder_forward_shapes() -> None:
    config = AutoencoderConfig(window_size=32, latent_dim=4)
    model = Conv1dAutoencoder(config)
    inputs = torch.randn(2, 5, 32)
    latent = model.encode(inputs)
    assert latent.shape == (2, 4)
    assert model(inputs).shape == inputs.shape


def test_training_artifact_and_residual_enhancer(tiny_model_artifact) -> None:
    assert (tiny_model_artifact / "model.pt").exists()
    assert (tiny_model_artifact / "latent_stats_by_condition.json").exists()
    assert (tiny_model_artifact / "latent_stats_by_condition.npz").exists()
    scenario = ScenarioConfig.from_file("examples/example_scenario_gas_sand.yaml")
    scenario.realism.mode = "none"
    well = generate_well(scenario)
    truth_snapshot = {
        name: deepcopy(getattr(well.truth, name))
        for name in (
            "facies",
            "lithology",
            "fluid",
            "vsh",
            "phi",
            "sw",
            "is_reservoir",
            "is_pay",
            "bad_hole_mask",
            "contacts",
        )
    }
    scenario.realism.mode = "autoencoder_mcmc"
    scenario.realism.model_path = str(tiny_model_artifact)
    scenario.realism.max_attempts_per_window = 3
    enhancer = AutoencoderMCMCRealismEnhancer(tiny_model_artifact)
    enhanced = enhancer.enhance(
        well.curves,
        well.truth,
        scenario,
        np.random.default_rng(11),
    )
    assert list(enhanced) == list(well.curves)
    assert len(enhanced) == len(well.curves)
    assert np.array_equal(enhanced["DEPT"], well.curves["DEPT"])
    for name, values in truth_snapshot.items():
        current = getattr(well.truth, name)
        assert current == values if isinstance(values, dict) else np.array_equal(current, values)
    assert enhancer.last_report["mode"] == "autoencoder_mcmc"
    assert enhancer.last_report["sampling_strategy"] == "condition_aware_metropolis_like"


def test_missing_model_uses_configured_fallback() -> None:
    scenario = ScenarioConfig.from_file("examples/example_scenario_gas_sand.yaml")
    scenario.realism.mode = "none"
    well = generate_well(scenario)
    scenario.realism.mode = "autoencoder_mcmc"
    scenario.realism.fallback = "none"
    enhancer = AutoencoderMCMCRealismEnhancer("missing-model")
    output = enhancer.enhance(
        well.curves,
        well.truth,
        scenario,
        np.random.default_rng(1),
    )
    assert output.equals(well.curves)
    assert enhancer.last_report["mode"] == "fallback"
