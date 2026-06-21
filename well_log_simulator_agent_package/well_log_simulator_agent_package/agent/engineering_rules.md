# Инженерные правила реализации

## Архитектурные правила

1. Core generator не должен зависеть от FastAPI, React или базы данных.
2. Любой генератор/экспортер/проверяющий модуль должен быть заменяемым.
3. Ground truth не должен восстанавливаться из финальных кривых; он должен генерироваться явно.
4. LAS — это выходной артефакт, а не внутренний формат данных.
5. Внутренний формат — Python dataclasses/Pydantic models + pandas/NumPy arrays.
6. ML-модули подключаются как optional extras.
7. Все случайные операции должны принимать seed.

## Suggested package structure

```text
synthetic_well_logs/
  cli.py
  config/scenario.py
  domain/{well.py,truth.py,curves.py,intervals.py}
  facies/{base.py,semi_markov.py,thickness.py}
  petrophysics/{truth_generator.py,distributions.py,correlations.py}
  forward/{base.py,gr.py,density.py,neutron.py,sonic.py,resistivity.py,caliper.py}
  realism/{base.py,noop.py,statistical.py,autoencoder_mcmc.py}
  constraints/{physics_constraints.py,range_checks.py}
  artifacts/{noise.py,washout.py,gaps.py,spikes.py,depth_shift.py}
  export/{las_exporter.py,truth_exporter.py,manifest_exporter.py,preview_exporter.py}
  datasets/{import_las.py,normalize_curves.py,qc.py,windows.py,parquet_store.py}
  checker/{interval_overlap.py,lithology_score.py,petrophysical_score.py}
```

## Coding conventions

- Python 3.11+.
- Type hints required.
- Pydantic для config models.
- `ruff` для форматирования/линтинга.
- `pytest` для тестов.
- Не смешивать domain logic и API route handlers.
- Не хардкодить curve mnemonics без слоя aliases/canonicalization.
