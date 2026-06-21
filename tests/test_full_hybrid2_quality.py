from __future__ import annotations

import json
from pathlib import Path

import lasio
import numpy as np

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.constraints import PhysicsConstraints
from synthetic_well_logs.constraints.scoring import TruthSlice, score_curves_against_truth
from synthetic_well_logs.datasets import CurveAliasesConfig, IngestionPipeline
from synthetic_well_logs.ml.constrained_search import ConstrainedLatentSearch
from synthetic_well_logs.ml.latent_sampling import (
    ConditionAwareLatentSampler,
    LatentDistribution,
    RealismCondition,
)
from synthetic_well_logs.realism import AutoencoderMCMCRealismEnhancer
from synthetic_well_logs.validation.statistical_gate import evaluate_statistical_gate


def _condition(facies: str, fluid: str = "water") -> RealismCondition:
    return RealismCondition(
        facies=np.array([facies] * 8),
        lithology=np.array(["sandstone"] * 8),
        fluid=np.array([fluid] * 8),
        target_curves=["GR", "RHOB", "NPHI", "DT", "RT"],
        depth=np.arange(8, dtype=float),
        difficulty="intermediate",
        dominant_facies=facies,
        dominant_lithology="sandstone",
        dominant_fluid=fluid,
        facies_fractions={facies: 1.0},
        has_pay=fluid != "water",
        has_gas=fluid == "gas",
    )


def test_condition_aware_sampler_selects_distinct_distributions() -> None:
    dimension = 4
    covariance = np.eye(dimension) * 1e-8
    sampler = ConditionAwareLatentSampler(
        np.full(dimension, -5.0),
        covariance,
        {
            "high_gr_shale_like": LatentDistribution(np.zeros(dimension), covariance, 100),
            "low_gr_clean_sand_like": LatentDistribution(np.full(dimension, 5.0), covariance, 100),
        },
        min_condition_count=30,
    )
    rng = np.random.default_rng(3)
    shale = sampler.sample(_condition("shale"), rng)
    clean = sampler.sample(_condition("clean_sandstone"), rng)
    unknown = sampler.sample(_condition("unknown"), rng)
    assert abs(float(shale.mean())) < 0.1
    assert float(clean.mean()) > 4.9
    assert float(unknown.mean()) < -4.9


def test_metropolis_like_search_improves_or_preserves_score() -> None:
    initial = np.full(4, 3.0)
    search = ConstrainedLatentSearch(proposal_scale=0.5).search(
        initial,
        _condition("shale"),
        decode_fn=lambda value: value,
        score_fn=lambda value: float(np.mean(value**2)),
        rng=np.random.default_rng(4),
        max_steps=80,
        temperature=0.05,
    )
    assert search.best_score <= float(np.mean(initial**2))
    assert search.steps == 80


def test_ingestion_rejects_missing_model_channels(tmp_path: Path) -> None:
    las = lasio.LASFile()
    depth = 1000 + np.arange(128) * 0.1
    las.well.WELL = "INCOMPLETE"
    las.append_curve("DEPT", depth, unit="m")
    las.append_curve("GR", np.full(128, 70.0), unit="API")
    las.append_curve("RHOB", np.full(128, 2.4), unit="g/cc")
    las.append_curve("NPHI", np.full(128, 0.2), unit="v/v")
    source = tmp_path / "incomplete.las"
    las.write(str(source), version=2.0)
    out = tmp_path / "calibration"
    pipeline = IngestionPipeline(CurveAliasesConfig.from_file(), window_size=32, stride=16)
    try:
        pipeline.ingest(source, out)
    except ValueError as exc:
        assert "no calibration windows" in str(exc)
    report = json.loads((out / "ingestion_report.json").read_text(encoding="utf-8"))
    rejected = report["files"][0]
    assert rejected["found_curves"] == ["GR", "RHOB", "NPHI"]
    assert "missing required model channels: DT, RT" in rejected["reason"]


def test_statistical_gate_passes_matching_and_fails_bad_statistics() -> None:
    curves = {
        name: {
            "count": 100,
            "mean": mean,
            "std": std,
            "p05": mean - std,
            "p50": mean,
            "p95": mean + std,
            "autocorrelation_lag1": 0.8,
            "range_violation_rate": 0.0,
        }
        for name, mean, std in (("GR", 80.0, 20.0), ("RHOB", 2.4, 0.1), ("RT", 10.0, 5.0))
    }
    correlation = {
        "GR": {"GR": 1.0, "RHOB": 0.2, "RT": -0.3},
        "RHOB": {"GR": 0.2, "RHOB": 1.0, "RT": -0.1},
        "RT": {"GR": -0.3, "RHOB": -0.1, "RT": 1.0},
    }
    real = {"curves": curves, "correlation_matrix": correlation}
    assert evaluate_statistical_gate(real, real)["status"] == "passed"
    bad_curves = {name: dict(values) for name, values in curves.items()}
    for values in bad_curves.values():
        values["mean"] = float(values["mean"]) + 10 * float(values["std"])
        values["std"] = 1e-6
        values["autocorrelation_lag1"] = 0.0
    bad = {"curves": bad_curves, "correlation_matrix": correlation}
    failed = evaluate_statistical_gate(real, bad)
    assert failed["status"] == "failed"
    assert failed["valid"] is False


def test_unified_score_and_constraints_detect_same_violations() -> None:
    scenario = ScenarioConfig.from_file("examples/example_scenario_gas_sand.yaml")
    scenario.realism.mode = "none"
    scenario.artifacts.noise = False
    scenario.artifacts.washout = False
    scenario.artifacts.spikes = False
    well = generate_well(scenario)
    corrupted = well.curves.copy()
    corrupted["GR"] = 250.0
    corrupted.loc[well.truth.fluid == "gas", "NPHI"] = 0.8
    corrupted.loc[well.truth.is_pay, "RT"] = 0.1
    truth_slice = TruthSlice.from_ground_truth(well.truth)
    before = score_curves_against_truth(
        {curve: corrupted[curve].to_numpy() for curve in ("GR", "RHOB", "NPHI", "DT", "RT")},
        truth_slice,
    )
    _, report = PhysicsConstraints().apply(corrupted, well.truth, scenario)
    assert before.components["gr_vsh_violation"] > 0
    assert before.components["gas_effect_violation"] > 0
    assert before.components["pay_interval_violation"] > 0
    assert report["score_components_before"]["gr_vsh_violation"] > 0
    assert report["score_components_before"]["gas_effect_violation"] > 0
    assert report["score_components_before"]["pay_interval_violation"] > 0


def test_realism_strength_and_fallback_quality(tiny_model_artifact: Path) -> None:
    training_report = json.loads(
        (tiny_model_artifact / "training_report.json").read_text(encoding="utf-8")
    )
    assert training_report["condition_statistics"]
    scenario = ScenarioConfig.from_file("examples/example_scenario_gas_sand.yaml")
    scenario.realism.mode = "none"
    scenario.artifacts.noise = False
    scenario.artifacts.washout = False
    scenario.artifacts.spikes = False
    base = generate_well(scenario)

    def enhanced(strength: float, threshold: float = 1.0):
        configured = scenario.model_copy(deep=True)
        configured.realism.mode = "autoencoder_mcmc"
        configured.realism.model_path = str(tiny_model_artifact)
        configured.realism.strength = strength
        configured.realism.max_constraint_score = threshold
        configured.realism.max_attempts_per_window = 1
        configured.realism.mcmc_steps_per_window = 1
        configured.realism.mcmc_proposal_scale = 1e-6
        configured.realism.mcmc_temperature = 1.0
        configured.realism.min_condition_count = 1
        enhancer = AutoencoderMCMCRealismEnhancer(tiny_model_artifact)
        output = enhancer.enhance(
            base.curves,
            base.truth,
            configured,
            np.random.default_rng(9),
        )
        return output, enhancer.last_report

    low, _ = enhanced(0.1)
    high, high_report = enhanced(0.8)
    low_residual = float(np.mean(np.abs(low["GR"] - base.curves["GR"])))
    high_residual = float(np.mean(np.abs(high["GR"] - base.curves["GR"])))
    assert high_residual > low_residual
    assert np.array_equal(high["DEPT"], base.curves["DEPT"])
    assert high_report["sampling_strategy"] == "condition_aware_metropolis_like"
    assert high_report["condition_usage"]
    assert high_report["residual_summary"]

    _, fallback_report = enhanced(0.4, threshold=0.0)
    assert fallback_report["fallback_windows"] > 0
    assert fallback_report["fallback_reasons"]
