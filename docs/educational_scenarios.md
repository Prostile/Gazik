# Учебные сценарии

| Сценарий | Required intervals | Ожидаемый тип ответа | Проверяемая ошибка |
|---|---|---|---|
| `coal_bed_detection` | `coal`, `siltstone` | lithology + facies | Уголь принят за pay |
| `shaly_sand_vs_siltstone` | `siltstone` | facies | Глинистый песчаник спутан с алевролитом |
| `carbonate_basic` | `marl`, `anhydrite` | lithology + facies | Ангидрит принят за коллектор, мергель за известняк |
| `bad_hole_quality_control` | `siltstone` | quality + reservoir/pay | Технический дефект принят за геологию |
| `gas_sand_vs_tight_sand` | `tight_sandstone` | facies + pay | Плотный песчаник принят за газовый pay |

`required_intervals` гарантируют, что ключевые нецелевые породы действительно
появятся, а не будут зависеть от случайного выбора фаций. Они вставляются вне target
interval, поэтому distractor не перезаписывает правильный продуктивный/водоносный
пласт.

Для каждого обязательного интервала можно задать `placement`:

```yaml
required_intervals:
  - facies: coal
    thickness_m: [2.0, 4.0]
    count: 1
    role: distractor
    placement: random
```

`placement: deterministic` сохраняет прежнее стабильное размещение в середине
доступной области и удобно для точных тестов. `placement: random` выбирает свободную
позицию через seed сценария, поэтому результат остается воспроизводимым, но становится
менее шаблонным.

В `truth.json` обязательные интервалы получают provenance-поля:

```json
{
  "facies": "coal",
  "injected_role": "distractor",
  "injection_source": "required_interval",
  "required_interval_index": 0,
  "placement": "random"
}
```

Эти поля нужны будущему интерфейсу преподавателя: он сможет показать, какие интервалы
были специально вставлены как distractor, comparison или quality-control зона.

Правильный ответ формируется из `truth.json`: маски `is_reservoir`, `is_pay`,
литология/фации и `artifacts`/`bad_hole_mask`. Водоносный целевой песчаник в первом
сценарии не является pay. Все сценарии лежат в `examples/educational/`.

```powershell
python -m synthetic_well_logs generate-educational-suite --out out/educational_suite
```
