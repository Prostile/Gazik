from pathlib import Path

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.student_answers import StudentAnswer, check_student_answer
from synthetic_well_logs.student_answers.truth_adapter import TruthData

ROOT = Path(__file__).parents[1]


def _well(name: str):
    scenario = ScenarioConfig.from_file(ROOT / "examples" / "educational" / name)
    return generate_well(scenario)


def _perfect_answer(well) -> StudentAnswer:
    truth = TruthData.from_source(well.truth)
    pay = []
    for interval in truth.mask_intervals(truth.is_pay):
        pay.append({**interval, "fluid": str(truth.fluid[truth.is_pay][0])})
    return StudentAnswer.model_validate(
        {
            "well_id": truth.well_id,
            "reservoir_intervals": truth.mask_intervals(truth.is_reservoir),
            "pay_intervals": pay,
            "lithology_intervals": [
                {
                    "top": item["top"],
                    "base": item["base"],
                    "lithology": item["lithology"],
                }
                for item in truth.intervals
            ],
            "bad_hole_intervals": truth.artifact_intervals(),
        }
    )


def test_perfect_empty_and_partial_answers() -> None:
    well = _well("bad_hole_quality_control.yaml")
    perfect = _perfect_answer(well)
    assert check_student_answer(perfect, well.truth).total_score >= 99
    assert check_student_answer(StudentAnswer(well_id=well.well_id), well.truth).total_score < 20
    partial = perfect.model_copy(deep=True)
    partial.reservoir_intervals = partial.reservoir_intervals[:1]
    score = check_student_answer(partial, well.truth).total_score
    assert 20 < score < 100


def test_coal_as_pay_and_anhydrite_as_reservoir_are_penalized() -> None:
    coal_well = _well("coal_bed_detection.yaml")
    coal_interval = next(item for item in coal_well.truth.intervals if item["facies"] == "coal")
    coal_answer = StudentAnswer.model_validate(
        {
            "well_id": coal_well.well_id,
            "pay_intervals": [
                {"top": coal_interval["top"], "base": coal_interval["base"], "fluid": "gas"}
            ],
        }
    )
    coal_report = check_student_answer(coal_answer, coal_well.truth)
    assert coal_report.pay_score["forbidden_lithology_predictions"]

    carbonate = _well("carbonate_basic.yaml")
    interval = next(item for item in carbonate.truth.intervals if item["facies"] == "anhydrite")
    answer = StudentAnswer.model_validate(
        {
            "well_id": carbonate.well_id,
            "reservoir_intervals": [{"top": interval["top"], "base": interval["base"]}],
        }
    )
    report = check_student_answer(answer, carbonate.truth)
    assert report.reservoir_score["forbidden_lithology_predictions"]


def test_missing_washout_is_reported() -> None:
    well = _well("bad_hole_quality_control.yaml")
    report = check_student_answer(StudentAnswer(well_id=well.well_id), well.truth)
    assert report.quality_score["missed_bad_hole_intervals"]
