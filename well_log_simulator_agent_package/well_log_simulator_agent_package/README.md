# Controlled Hybrid Synthetic Well Log Simulator — пакет для ИИ-агента

Дата подготовки: 2026-06-21.

Архив содержит документы для реализации симулятора синтетических well logs / ГИС для учебного тренажера.

Текущая целевая архитектура: **Hybrid #2 — Markov / Semi-Markov facies + Autoencoder/MCMC realism + Physics constraints**.

Будущий переход: **Hybrid #3 — Physics truth + GAN/Diffusion residual refinement + AI evaluator**.

## Состав архива

```text
docs/
  01_technical_spec.md
  02_architecture_decision_record.md
  03_generator_core_spec.md
  04_realism_layer_spec.md
  05_data_ingestion_and_datasets.md
  06_validation_and_acceptance.md
  07_api_and_cli_spec.md
  08_implementation_plan.md
agent/
  ai_agent_task.md
  engineering_rules.md
research/
  research_papers_digest.md
  bibliography.bib
  source_map.md
diagrams/
  01_overall_architecture.mmd
  02_generation_pipeline.mmd
  03_module_interfaces.mmd
  04_data_flow.mmd
  05_future_v3_architecture.mmd
schemas/
  scenario_schema.json
  ground_truth_schema.json
  generated_well_manifest_schema.json
  student_submission_schema.json
examples/
  example_scenario_gas_sand.yaml
  example_scenario_shaly_sand.yaml
  example_ground_truth.json
  example_generated_well_manifest.json
configs/
  requirements_mvp.txt
  requirements_research.txt
  pyproject_suggestion.toml
backlog/
  implementation_backlog.md
  risks_and_open_questions.md
```

## Главный принцип

ML-модуль может улучшать текстуру и статистический реализм кривых, но **не должен изменять скрытую геологическую истину**.

```text
ground_truth is authoritative
ML realism is a constrained post-processing / calibration layer
```

## Минимальный результат первой реализации

ИИ-агент должен реализовать Python core, который по YAML-сценарию генерирует:

```text
well_001.las
well_001_truth.json
well_001_manifest.json
well_001_preview.html or .png
```

Команда:

```bash
python -m synthetic_well_logs generate \
  --scenario examples/example_scenario_gas_sand.yaml \
  --out data/generated/well_001
```
