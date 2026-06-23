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

## What “Hybrid #2” means in this project

Hybrid #2 — не чистый нейросетевой генератор. Это truth-preserving simulator с
data-driven residual realism layer. Геологический случай и правильный учебный ответ
задаются scenario config, Semi-Markov stratigraphy и petrophysical truth. Autoencoder
не создаёт геологию: он изучает локальную residual texture calibration LAS и
накладывает её только на наблюдаемые кривые.

Calibration и autoencoder используют строгий пятиканальный контракт:
`GR`, `RHOB`, `NPHI`, `DT`, `RT`. LAS без любого из этих каналов отклоняется с
явным ingestion report. Variable-channel autoencoder в текущую версию не входит.

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

## Educational scenarios and student answer checking

Пять готовых учебных задач генерируются одной командой. Сценарии используют
`required_intervals`, поэтому учебные distractor/comparison породы гарантированно
присутствуют в hidden truth:

```powershell
python -m synthetic_well_logs generate-educational-suite --out out/educational_suite
```

Проверка ответа студента и структуры будущей коллекции реальных данных:

```powershell
python -m synthetic_well_logs check-answer `
  --answer answers/student_001.json `
  --truth out/well_001_truth.json `
  --out reports/student_001_check.json

python -m synthetic_well_logs validate-dataset-structure --root data/raw_las
```

Ответы студентов могут содержать как `lithology_intervals`, так и более детальные
`facies_intervals`; старый формат без `facies_intervals` остается валидным. Форматы
описаны в `docs/student_answer_format.md` и `docs/data_collection_guidelines.md`.

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
`latent_stats.npz`, condition-aware `latent_stats_by_condition.*` и
`training_report.json`. Calibration windows получают heuristic electrofacies labels;
runtime sampler выбирает соответствующую latent distribution по facies, lithology,
fluid/pay context и difficulty, либо использует global distribution.

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

### MCMC-like vs full MCMC

`AutoencoderMCMCRealismEnhancer` реализует condition-aware Metropolis-like constrained
latent search: initial proposal → decode → residual → unified physics score → локальные
Markov proposals → accept/reject → лучший candidate. Это не полноценный Bayesian MCMC,
не posterior inference и не probabilistic inversion. Название сохранено как контракт
архитектуры Hybrid #2, а точная стратегия публикуется в manifest.

## Validation and reports

Manifest содержит pre-artifact constraints, post-artifact validation,
educational validation и сведения о применённом realism model/fallback.

Сравнение synthetic LAS с calibration dataset:

```powershell
.venv\Scripts\python -m synthetic_well_logs compare-stats `
  --synthetic out/hybrid2_well_001.las `
  --calibration data/calibration `
  --out reports/hybrid2_well_001 `
  --strict
```

Отчёт включает distributions, percentiles, missing/range rates, correlations,
lag-1 autocorrelation и crossplots RHOB–NPHI, GR–RT, RHOB–DT. Statistical gate
возвращает `passed`, `warning` или `failed`; `--strict` завершает команду кодом 1
для `failed`. Если в scenario включены `statistical_gate: true` и
`calibration_dataset_path`, gate summary также попадает в manifest.

## Финальный workflow Hybrid #2

```powershell
.venv\Scripts\python -m synthetic_well_logs ingest-las `
  --input data/raw_las --out data/calibration

.venv\Scripts\python -m synthetic_well_logs train-autoencoder `
  --dataset data/calibration --out models/autoencoder_v001

.venv\Scripts\python -m synthetic_well_logs generate `
  --scenario examples/example_scenario_autoencoder_mcmc.yaml `
  --out out/hybrid2_well_001

.venv\Scripts\python -m synthetic_well_logs compare-stats `
  --synthetic out/hybrid2_well_001.las `
  --calibration data/calibration `
  --out reports/hybrid2_well_001 --strict
```

Manifest фиксирует condition usage, accepted/rejected/fallback windows, MCMC steps,
constraint score quantiles, fallback reasons, residual magnitudes и statistical gate.

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
- Текущий MCMC — Metropolis-like constrained latent search, а не полноценный
  Bayesian posterior sampler или метод инверсии.
- Качество ML realism зависит от объёма и однородности calibration LAS.
- GAN, diffusion, AI evaluator, FastAPI и production frontend относятся к будущему
  Hybrid #3 и в этот пакет не входят.

Исходные спецификации сохранены в `well_log_simulator_agent_package/`.
