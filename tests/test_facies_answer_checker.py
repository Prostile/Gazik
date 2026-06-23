from pathlib import Path

from synthetic_well_logs import ScenarioConfig, generate_well
from synthetic_well_logs.student_answers import StudentAnswer, check_student_answer
from synthetic_well_logs.student_answers.truth_adapter import TruthData

ROOT = Path(__file__).parents[1]


def _well(name: str):
    scenario = ScenarioConfig.from_file(ROOT / "examples" / "educational" / name)
    return generate_well(scenario)


def _facies_answer(well, replacement: dict[str, str] | None = None) -> StudentAnswer:
    truth = TruthData.from_source(well.truth)
    replacement = replacement or {}
    return StudentAnswer.model_validate(
        {
            "well_id": well.well_id,
            "facies_intervals": [
                {
                    "top": item["top"],
                    "base": item["base"],
                    "facies": replacement.get(item["facies"], item["facies"]),
                }
                for item in truth.intervals
            ],
        }
    )


def test_perfect_facies_answer_scores_high() -> None:
    well = _well("shaly_sand_vs_siltstone.yaml")
    report = check_student_answer(_facies_answer(well), well.truth)
    assert report.facies_score is not None
    assert report.facies_score["score"] >= 99


def test_shaly_sandstone_vs_siltstone_confusion_is_penalized() -> None:
    well = _well("shaly_sand_vs_siltstone.yaml")
    report = check_student_answer(
        _facies_answer(well, {"shaly_sandstone": "siltstone"}),
        well.truth,
    )
    assert report.facies_score is not None
    assert report.facies_score["score"] < 95
    assert report.facies_score["major_confusions"]


def test_tight_sandstone_as_clean_sandstone_is_penalized() -> None:
    well = _well("gas_sand_vs_tight_sand.yaml")
    report = check_student_answer(
        _facies_answer(well, {"tight_sandstone": "clean_sandstone"}),
        well.truth,
    )
    assert report.facies_score is not None
    assert report.facies_score["score"] < 100
    assert report.facies_score["major_confusions"]


def test_old_student_answer_without_facies_still_works() -> None:
    well = _well("shaly_sand_vs_siltstone.yaml")
    report = check_student_answer(StudentAnswer(well_id=well.well_id), well.truth)
    assert report.facies_score is None

