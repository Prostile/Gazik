from synthetic_well_logs import ScenarioConfig, generate_well


def test_extended_constraint_and_post_artifact_reports() -> None:
    scenario = ScenarioConfig.from_file("examples/example_scenario_gas_sand.yaml")
    well = generate_well(scenario)
    physical = well.validation["pre_artifact_constraints"]
    assert physical["constraint_violation_rate"] <= 0.05
    assert physical["pay_interval_preserved"] is True
    assert physical["gas_effect_preserved"] is True
    assert "gr_vsh_violation_rate" in physical
    assert "rhob_phi_violation_rate" in physical
    assert "rt_sw_violation_rate" in physical
    post = well.validation["post_artifact_validation"]
    assert post["valid"] is True
    assert post["artifact_provenance_preserved"] is True
    assert post["depth_unchanged"] is True
