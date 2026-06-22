# Учебные сценарии

| Сценарий | Цель | Ключевые кривые | Проверяемая ошибка |
|---|---|---|---|
| `coal_bed_detection` | Найти уголь | RHOB, NPHI, RT, GR | Уголь принят за pay |
| `shaly_sand_vs_siltstone` | Разделить глинистый песчаник и алевролит | GR, RHOB, RT | Завышена эффективная толщина |
| `carbonate_basic` | Различить limestone, dolomite, marl, anhydrite | GR, RHOB, NPHI, DT | Ангидрит принят за коллектор |
| `bad_hole_quality_control` | Найти washout и пропуски | CALI, RHOB, NPHI | Технический дефект принят за геологию |
| `gas_sand_vs_tight_sand` | Отличить газовый от плотного песчаника | RHOB–NPHI, RT, GR | Плотный песчаник принят за pay |

Правильный ответ формируется из `truth.json`: маски `is_reservoir`, `is_pay`,
литология/фации и `artifacts`/`bad_hole_mask`. Водоносный целевой песчаник в первом
сценарии не является pay. Все сценарии лежат в `examples/educational/`.

```powershell
python -m synthetic_well_logs generate-educational-suite --out out/educational_suite
```
