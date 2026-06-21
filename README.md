# Synthetic Well Logs Simulator — Hybrid #2

Управляемый генератор учебных LAS-файлов с известным hidden ground truth,
физической forward-моделью и data-driven residual realism layer.

```text
Semi-Markov facies → petrophysical truth → physics logs
→ tool-resolution smoothing → realism enhancer → physics constraints
→ artifacts → post-artifact validation → LAS/truth/manifest/preview
```

Главный инвариант: realism layer изменяет только наблюдаемые кривые. Facies,
lithology, fluid, Vsh, PHI, Sw, reservoir/pay flags и contacts формируются раньше
и остаются authoritative ground truth.

## Установка

Базовая генерация без ML:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Полный Hybrid #2 с Parquet и PyTorch:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,hybrid2]"
```

## Quickstart

```powershell
.venv\Scripts\python -m synthetic_well_logs generate `
  --scenario examples/example_scenario_gas_sand.yaml `
  --out out/well_001

.venv\Scripts\python -m synthetic_well_logs validate `
  --well out/well_001.las `
  --truth out/well_001_truth.json
```

Результат: `well_001.las`, `well_001_truth.json`, `well_001_manifest.json` и
интерактивный `well_001_preview.html`.

## Data ingestion

Pipeline читает реальные LAS, canonicalize-ит aliases, нормализует units,
ресэмплирует depth без заполнения крупных gaps, строит QC-маски и окна.

```powershell
.venv\Scripts\python -m synthetic_well_logs ingest-las `
  --input data/raw_las `
  --out data/calibration `
  --aliases configs/curve_aliases.yaml `
  --target-step-m 0.1 `
  --window-size 128 `
  --stride 64
```

Dataset содержит `curves.parquet`, `metadata.parquet`, `windows/*.npz`,
`manifest.json`, `ingestion_report.json` и `statistics_report.json`.

## Training calibration autoencoder

```powershell
.venv\Scripts\python -m synthetic_well_logs train-autoencoder `
  --dataset data/calibration `
  --out models/autoencoder_v001 `
  --epochs 20 `
  --latent-dim 32
```

Artifact состоит из `model.pt`, `config.json`, `normalization.json`,
`latent_stats.npz` и `training_report.json`.

## Generation with autoencoder_mcmc

Укажите artifact в `realism.model_path` либо обучите его по пути из примера:

```powershell
.venv\Scripts\python -m synthetic_well_logs generate `
  --scenario examples/example_scenario_autoencoder_mcmc.yaml `
  --out out/hybrid2_well_001
```

Enhancer сэмплирует latent vectors, декодирует локальную texture, извлекает
high-pass residual и накладывает его на physics curves через overlap-add. Для RT
используется multiplicative residual. Не прошедшие constraint score окна переходят
на настроенный `statistical` или `none` fallback.

## Validation and reports

Manifest содержит pre-artifact constraints, post-artifact validation,
educational validation и сведения о применённом realism model/fallback.

Сравнение synthetic LAS с calibration dataset:

```powershell
.venv\Scripts\python -m synthetic_well_logs compare-stats `
  --synthetic out/hybrid2_well_001.las `
  --calibration data/calibration `
  --out reports/hybrid2_well_001
```

Отчёт включает distributions, percentiles, missing/range rates, correlations,
lag-1 autocorrelation и crossplots RHOB–NPHI, GR–RT, RHOB–DT.

## Python API

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

## Ограничения

- Carbonate petrophysics остаётся упрощённой.
- Constrained latent sampling — rejection sampling по Gaussian latent model, а
  не полноценный Markov-chain Monte Carlo.
- Качество ML realism зависит от объёма и однородности calibration LAS.
- GAN, diffusion, AI evaluator, FastAPI и production frontend относятся к будущему
  Hybrid #3 и в этот пакет не входят.

Исходные спецификации сохранены в `well_log_simulator_agent_package/`.
