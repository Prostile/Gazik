# Формат ответа студента

Ответ — JSON-объект с обязательным `well_id` и четырьмя списками. Сами списки можно
оставить пустыми. Каждый интервал задается парой `top`, `base` (`base > top`) и
необязательной уверенностью от 0 до 1.

```json
{
  "well_id": "TRAINING_GAS_SAND_VS_TIGHT_SAND_001",
  "reservoir_intervals": [{"top": 2140.0, "base": 2150.0, "confidence": 0.8}],
  "pay_intervals": [{"top": 2142.0, "base": 2148.0, "fluid": "gas"}],
  "lithology_intervals": [{"top": 2100.0, "base": 2110.0, "lithology": "shale"}],
  "bad_hole_intervals": [{"top": 2190.0, "base": 2195.0, "reason": "washout"}]
}
```

`fluid`: `water`, `oil`, `gas` или `mixed`. `reason`: `washout`,
`missing_interval`, `spikes`, `depth_shift` или `other`. Внутренние названия пород
английские; допустимы как общая литология (`sandstone`), так и название фации.

Интервалы reservoir/pay и дефектов сопоставляются по intersection-over-union.
Литология оценивается по каждому отсчету глубины. Итог: reservoir 30%, pay 30%,
lithology 25%, quality 15%. Уголь/ангидрит как reservoir или pay получают отдельный
штраф. В отчете значения 0–1 — доли, `score` и `total_score` — баллы 0–100.

```powershell
python -m synthetic_well_logs check-answer `
  --answer answers/student_001.json `
  --truth out/well_001_truth.json `
  --out reports/student_001_check.json
```
