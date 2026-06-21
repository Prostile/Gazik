import json
from pathlib import Path

import lasio

from synthetic_well_logs import ScenarioConfig, generate_well


def test_hybrid2_end_to_end(tmp_path: Path, tiny_model_artifact: Path) -> None:
    scenario = ScenarioConfig.from_file("examples/example_scenario_autoencoder_mcmc.yaml")
    scenario.realism.model_path = str(tiny_model_artifact)
    scenario.realism.max_attempts_per_window = 3
    well = generate_well(scenario)
    paths = well.export(tmp_path / "hybrid2")
    las = lasio.read(paths["las"])
    truth = json.loads(paths["truth"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert len(las["DEPT"]) == len(truth["samples"]["depth"])
    validation = manifest["validation"]
    assert validation["pre_artifact_constraints"]["constraint_violation_rate"] <= 0.05
    assert validation["post_artifact_validation"]["valid"] is True
    assert validation["statistical"]["mode"] == "autoencoder_mcmc"
