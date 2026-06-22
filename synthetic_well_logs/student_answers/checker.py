from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from synthetic_well_logs.domain import GroundTruth
from synthetic_well_logs.student_answers.lithology_score import score_lithology
from synthetic_well_logs.student_answers.pay_score import score_pay
from synthetic_well_logs.student_answers.quality_score import score_quality
from synthetic_well_logs.student_answers.reports import StudentCheckReport
from synthetic_well_logs.student_answers.reservoir_score import score_reservoirs
from synthetic_well_logs.student_answers.schema import StudentAnswer
from synthetic_well_logs.student_answers.truth_adapter import TruthData


def check_student_answer(
    answer: StudentAnswer,
    truth: GroundTruth | Mapping[str, Any],
) -> StudentCheckReport:
    data = TruthData.from_source(truth)
    if data.well_id and answer.well_id != data.well_id:
        raise ValueError(
            f"answer well_id {answer.well_id!r} does not match truth well_id {data.well_id!r}"
        )
    reservoir = score_reservoirs(answer.reservoir_intervals, data)
    pay = score_pay(answer.pay_intervals, data)
    lithology = score_lithology(answer.lithology_intervals, data)
    quality = score_quality(answer.bad_hole_intervals, data)
    total = (
        0.30 * reservoir["score"]
        + 0.30 * pay["score"]
        + 0.25 * lithology["score"]
        + 0.15 * quality["score"]
    )
    feedback: list[str] = []
    if reservoir["f1"] >= 0.9:
        feedback.append("Коллекторы выделены точно.")
    elif reservoir["missed_intervals"]:
        feedback.append("Часть коллекторов пропущена.")
    if pay["forbidden_lithology_predictions"]:
        feedback.append("Уголь или ангидрит ошибочно отнесен к продуктивному интервалу.")
    elif pay["pay_f1"] >= 0.9:
        feedback.append("Продуктивные интервалы выделены точно.")
    if lithology["major_confusions"]:
        feedback.append("Есть существенные ошибки определения литологии.")
    if quality["missed_bad_hole_intervals"]:
        feedback.append("Часть интервалов плохого качества записи пропущена.")
    if not feedback:
        feedback.append("Ответ проверен; изучите метрики по каждому разделу.")
    return StudentCheckReport(
        well_id=answer.well_id,
        total_score=round(float(total), 2),
        reservoir_score=reservoir,
        pay_score=pay,
        lithology_score=lithology,
        quality_score=quality,
        feedback=feedback,
    )

