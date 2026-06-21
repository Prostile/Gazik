# API and CLI specification

## 1. CLI first

Первую реализацию делать через CLI.

### Generate command

```bash
python -m synthetic_well_logs generate \
  --scenario examples/example_scenario_gas_sand.yaml \
  --out data/generated/well_001
```

Output:

```text
data/generated/well_001/well.las
data/generated/well_001/truth.json
data/generated/well_001/manifest.json
data/generated/well_001/preview.html
```

### Validate command

```bash
python -m synthetic_well_logs validate \
  --well data/generated/well_001/well.las \
  --truth data/generated/well_001/truth.json
```

### Import LAS command

```bash
python -m synthetic_well_logs import-las \
  --input data/raw/nlog/*.las \
  --out data/processed/nlog.parquet
```

## 2. Future FastAPI endpoints

```http
POST /wells/generate
GET  /wells/{well_id}
GET  /wells/{well_id}/download-las
GET  /wells/{well_id}/truth
POST /tasks
POST /tasks/{task_id}/submit
```

## 3. Python API

```python
from synthetic_well_logs import generate_well
from synthetic_well_logs.config import ScenarioConfig

scenario = ScenarioConfig.from_yaml("scenario.yaml")
well = generate_well(scenario)
well.export("out/well_001")
```
