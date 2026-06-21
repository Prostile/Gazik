from pathlib import Path

import numpy as np

from synthetic_well_logs import ScenarioConfig, generate_well

ROOT = Path(__file__).parents[1]


def _scenario(name: str) -> ScenarioConfig:
    return ScenarioConfig.from_file(ROOT / "examples" / name)


def test_generation_is_deterministic() -> None:
    scenario = _scenario("example_scenario_gas_sand.yaml")
    first = generate_well(scenario)
    second = generate_well(scenario)
    assert first.curves.equals(second.curves)
    assert np.array_equal(first.truth.facies, second.truth.facies)
    assert np.allclose(first.truth.phi, second.truth.phi)
    assert first.truth.artifacts == second.truth.artifacts


def test_gas_sand_learning_objective_is_present() -> None:
    scenario = _scenario("example_scenario_gas_sand.yaml")
    well = generate_well(scenario)
    pay = well.truth.is_pay
    assert pay.any()
    assert np.all(well.truth.facies[pay] == "clean_sandstone")
    assert np.all(well.truth.fluid[pay] == "gas")
    assert float(well.curves.loc[pay, "RT"].median()) > 2.0
    assert float(well.curves.loc[pay, "NPHI"].median()) < float(
        well.truth.phi[pay].mean()
    )
    pay_thickness = pay.sum() * scenario.depth.step
    low, high = scenario.target.net_pay_thickness_m
    assert low <= pay_thickness <= high


def test_washout_updates_curves_and_truth_mask() -> None:
    well = generate_well(_scenario("example_scenario_shaly_sand.yaml"))
    mask = well.truth.bad_hole_mask
    assert mask.any()
    assert float(well.curves.loc[mask, "CALI"].mean()) > 10.0
    assert any(artifact["type"] == "washout" for artifact in well.truth.artifacts)
    assert any(artifact["type"] == "missing_interval" for artifact in well.truth.artifacts)
    assert well.curves.isna().any().any()


def test_curve_ranges_after_constraints() -> None:
    well = generate_well(_scenario("example_scenario_gas_sand.yaml"))
    assert well.curves["GR"].dropna().between(0, 250).all()
    assert well.curves["RHOB"].dropna().between(1.5, 3.1).all()
    assert well.curves["NPHI"].dropna().between(-0.15, 0.8).all()
    assert well.curves["RT"].dropna().between(0.1, 10_000).all()
