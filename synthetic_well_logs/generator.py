from __future__ import annotations

import numpy as np

from synthetic_well_logs.artifacts import ArtifactSimulator
from synthetic_well_logs.config import ScenarioConfig
from synthetic_well_logs.constraints import PhysicsConstraints
from synthetic_well_logs.domain import GeneratedWell
from synthetic_well_logs.facies import SemiMarkovFaciesGenerator
from synthetic_well_logs.forward import PhysicsForwardLogModel
from synthetic_well_logs.petrophysics import PetrophysicalTruthGenerator
from synthetic_well_logs.realism import NoOpRealismEnhancer, StatisticalRealismEnhancer


def create_depth_grid(scenario: ScenarioConfig) -> np.ndarray:
    span = scenario.depth.stop - scenario.depth.start
    intervals = int(round(span / scenario.depth.step))
    if not np.isclose(intervals * scenario.depth.step, span, atol=1e-8):
        raise ValueError("depth range must be evenly divisible by depth.step")
    decimals = max(0, int(np.ceil(-np.log10(scenario.depth.step))) + 3)
    return np.round(
        scenario.depth.start + np.arange(intervals + 1) * scenario.depth.step,
        decimals=decimals,
    )


def generate_well(scenario: ScenarioConfig) -> GeneratedWell:
    rng = np.random.default_rng(scenario.seed)
    depth = create_depth_grid(scenario)
    facies, intervals, target_mask = SemiMarkovFaciesGenerator().generate(depth, scenario, rng)
    truth = PetrophysicalTruthGenerator().generate(
        depth,
        facies,
        intervals,
        target_mask,
        scenario,
        rng,
    )
    curves = PhysicsForwardLogModel().generate(truth, scenario, rng)

    if scenario.realism.mode == "none":
        enhancer = NoOpRealismEnhancer()
    else:
        enhancer = StatisticalRealismEnhancer()
    curves = enhancer.enhance(curves, truth, scenario, rng)
    curves, constraint_report = PhysicsConstraints().apply(curves, truth, scenario)
    curves = ArtifactSimulator().apply(curves, truth, scenario, rng)

    return GeneratedWell(
        well_id=scenario.well.name,
        depth=depth,
        curves=curves,
        truth=truth,
        scenario=scenario,
        validation=constraint_report,
    )
