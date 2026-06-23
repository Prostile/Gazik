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
  "facies_intervals": [{"top": 2110.0, "base": 2122.0, "facies": "shaly_sandstone"}],
  "bad_hole_intervals": [{"top": 2190.0, "base": 2195.0, "reason": "washout"}]
}
```

`fluid`: `water`, `oil`, `gas` или `mixed`. `reason`: `washout`,
`missing_interval`, `spikes`, `depth_shift` или `other`. Внутренние названия пород
английские.

`lithology_intervals` и `facies_intervals` проверяются отдельно:

- `lithology_intervals` — общая порода: `sandstone`, `shale`, `limestone`,
  `dolomite`, `coal`, `anhydrite`;
- `facies_intervals` — детальная учебная фация: `clean_sandstone`,
  `shaly_sandstone`, `tight_sandstone`, `siltstone`, `marl`.

Полный пример для задания на глинистый песчаник и алевролит:

```json
{
  "well_id": "TRAINING_SHALY_SAND_VS_SILTSTONE_001",
  "reservoir_intervals": [],
  "pay_intervals": [],
  "lithology_intervals": [],
  "facies_intervals": [
    {"top": 1210.0, "base": 1222.0, "facies": "shaly_sandstone"},
    {"top": 1222.0, "base": 1230.0, "facies": "siltstone"}
  ],
  "bad_hole_intervals": []
}
```

Интервалы reservoir/pay и дефектов сопоставляются по intersection-over-union.
Литология и фации оцениваются по каждому отсчету глубины. Если `facies_intervals`
нет, используются старые веса: reservoir 30%, pay 30%, lithology 25%, quality 15%.
Если `facies_intervals` есть, веса такие: reservoir 25%, pay 25%, lithology 20%,
facies 20%, quality 10%. Уголь/ангидрит как reservoir или pay получают отдельный
штраф. В отчете значения 0–1 — доли, `score` и `total_score` — баллы 0–100.

```powershell
python -m synthetic_well_logs check-answer `
  --answer answers/student_001.json `
  --truth out/well_001_truth.json `
  --out reports/student_001_check.json
```
