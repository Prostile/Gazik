# Synthetic Well Logs

Управляемый генератор синтетических каротажных диаграмм для учебных задач по
интерпретации ГИС. Геологическая истина создаётся до слоя реализма и не выводится
обратно из наблюдаемых кривых.

## Установка

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Генерация

```powershell
.venv\Scripts\python -m synthetic_well_logs generate `
  --scenario examples/example_scenario_gas_sand.yaml `
  --out out/well_001
```

Команда создаёт:

```text
out/well_001.las
out/well_001_truth.json
out/well_001_manifest.json
out/well_001_preview.html
```

Проверка результата:

```powershell
.venv\Scripts\python -m synthetic_well_logs validate `
  --well out/well_001.las `
  --truth out/well_001_truth.json
```

Python API:

```python
from synthetic_well_logs import ScenarioConfig, generate_well

scenario = ScenarioConfig.from_file("examples/example_scenario_gas_sand.yaml")
well = generate_well(scenario)
well.export("out/well_001")
```

## Проверки

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```

Исходный пакет спецификаций сохранён в `well_log_simulator_agent_package/`.

