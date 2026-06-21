# Задание для ИИ-агента

## Роль агента

Ты — инженер-разработчик и научный ассистент по Python/scientific computing. Твоя задача — реализовать ядро симулятора синтетических well logs / ГИС для учебного тренажера.

## Цель

Реализовать модульный генератор синтетических LAS-файлов с управляемым геологическим сценарием, скрытым ground truth и расширяемым realism layer.

Текущий целевой гибрид:

```text
ScenarioConfig
  → SemiMarkovFaciesGenerator
  → PetrophysicalTruthGenerator
  → PhysicsForwardLogModel
  → Autoencoder/MCMC RealismEnhancer
  → PhysicsConstraints
  → ArtifactSimulator
  → LAS Exporter
  → GroundTruth Exporter
```

## Обязательные требования

1. Генератор должен работать без ML-модуля.
2. ML-модуль должен быть подключаемым через интерфейс `IRealismEnhancer`.
3. Ground truth формируется до ML-модуля и остается главным источником истины.
4. LAS должен экспортироваться через `lasio`.
5. Все сценарии должны задаваться через YAML/JSON и валидироваться Pydantic-моделями.
6. Нужен CLI-режим генерации.
7. Нужны unit tests для основных модулей.
8. Нужны минимум два демонстрационных сценария: gas-bearing clean sandstone и shaly sandstone with washout.
9. Нельзя строить архитектуру так, чтобы GAN/diffusion стали обязательными для MVP.
10. Код должен быть расширяемым под будущий Hybrid #3.

## Приоритет реализации

### Этап A — базовый управляемый генератор

- Depth grid.
- Semi-Markov facies sequence.
- Bed thickness distributions.
- Hidden petrophysical truth: lithology, facies, Vsh, PHI, Sw, fluid, net reservoir, net pay.
- Forward logs: GR, RHOB, NPHI, DT, RT, CALI.
- Artifacts: noise, spikes, washout, missing intervals, depth shift.
- Export: LAS, ground_truth.json, manifest.json, preview plot.

### Этап B — data ingestion для реальных LAS

- Import LAS via `lasio`.
- Canonicalize curve mnemonics.
- Resample to regular depth step.
- QC missing values/outliers.
- Segment into windows for training/calibration.
- Save flattened/intermediate dataset to Parquet.

### Этап C — realism layer v1

- Implement `NoOpRealismEnhancer`.
- Implement `StatisticalRealismEnhancer`.
- Prepare interface for `AutoencoderMCMCRealismEnhancer`.

### Этап D — Autoencoder/MCMC prototype

- Train small autoencoder on normalized curve windows.
- Sample latent vectors conditioned by facies/electrofacies label if labels exist.
- Decode latent texture/residual.
- Blend with physics-generated curves.
- Apply physics constraints after blending.

## Definition of Done

Реализация считается готовой, если:

```bash
python -m synthetic_well_logs generate --scenario examples/example_scenario_gas_sand.yaml --out out/well_001
pytest
```

создает валидные:

```text
out/well_001.las
out/well_001_truth.json
out/well_001_manifest.json
out/well_001_preview.html
```
