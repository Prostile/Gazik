# Architecture Decision Record

## ADR-001: Выбор гибридной архитектуры

### Решение

Использовать гибрид:

```text
Markov / Semi-Markov Facies Generator
  + Hidden Petrophysical Truth
  + Physics Forward Log Model
  + Autoencoder/MCMC Realism Layer
  + Physics Constraints
```

### Почему не чистый GAN/diffusion

Чистые генеративные модели могут создать реалистичные кривые, но для учебного тренажера критично:

```text
- управлять сценарием;
- знать правильный ответ;
- объяснять, почему ответ правильный;
- гарантировать наличие заданных геологических событий;
- делать задания разной сложности.
```

GAN/diffusion оставляем как будущий residual realism layer.

### Почему нужен Markov / Semi-Markov

Well logs — вертикальные последовательности. Студент должен видеть геологически связную последовательность пластов. Semi-Markov лучше простого Markov по depth step, потому что позволяет явно моделировать толщины пластов.

### Почему нужен physics forward model

Hidden truth должен порождать кривые через понятные петрофизические зависимости. Это дает explainability, controllability, automatic checking и физическую непротиворечивость.

### Почему нужен Autoencoder/MCMC layer

Physics-only кривые могут быть слишком гладкими. Autoencoder/MCMC layer нужен для сохранения multivariate correlations, реалистичной локальной изменчивости, калибровки на реальные LAS и подготовки к будущим ML-модулям.

### Ограничение

ML-модуль не должен менять:

```text
facies
lithology
fluid
net reservoir
net pay
OWC/GOC
bad_hole intervals
```

Он может корректировать только наблюдаемые кривые в допустимых пределах.

## ADR-002: Внутренний формат данных

Внутренние данные: Pydantic/domain models + pandas DataFrame/NumPy arrays. LAS используется только как input/output artifact.

## ADR-003: Библиотека LAS

Использовать `lasio` для чтения/записи LAS.

## ADR-004: Web/API

На первом этапе core + CLI. FastAPI позже. Генератор должен быть независимым от backend/frontend.

## ADR-005: Будущая совместимость с Hybrid #3

Ввести интерфейсы:

```text
IRealismEnhancer
IAnswerEvaluator
ICurveImputer
```

Будущие реализации:

```text
DiffusionResidualEnhancer
TSGANImputer
FoundationModelAnswerEvaluator
```
