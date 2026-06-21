# Техническое задание: гибридный симулятор синтетических well logs / ГИС

## 1. Назначение

Разработать симулятор, который генерирует реалистичные управляемые учебные LAS-файлы для тренажера по интерпретации геофизических исследований скважин.

Продуктовая задача:

```text
Преподаватель задает сценарий → система генерирует LAS → студент интерпретирует → система сравнивает ответ с hidden ground truth.
```

## 2. Целевая архитектура

Текущая целевая версия:

```text
Hybrid #2:
Markov / Semi-Markov facies
  + petrophysical hidden truth
  + physics forward response
  + Autoencoder/MCMC realism calibration
  + physics constraints
```

Будущее развитие:

```text
Hybrid #3:
Physics truth
  + GAN/Diffusion residual refinement
  + AI evaluator
```

## 3. Scope первой реализации

### Входит

- YAML/JSON scenario config.
- Генерация depth grid.
- Генерация facies/stratigraphy sequence.
- Генерация hidden petrophysical truth.
- Расчет кривых: DEPT, GR, CALI, RHOB, NPHI, DT, RT.
- Artifact simulator: noise, spikes, washout, missing intervals, depth shift, vertical smoothing.
- LAS export.
- Ground truth JSON export.
- Manifest export.
- Preview plot.
- CLI.

### Не входит в первую реализацию

- Production frontend.
- Full learning management system.
- Полноценный GAN/diffusion.
- Сейсмическая инверсия.
- DLIS/LIS export.
- Полная карбонатная петрофизика.

## 4. Функциональные требования

### FR-1. Scenario config

Система должна принимать YAML/JSON config.

Минимальные поля:

```text
well.name
depth.start
depth.stop
depth.step
geology.depositional_environment
geology.stacking_pattern
target.reservoir_type
target.hydrocarbon
curves
artifacts
difficulty
seed
```

### FR-2. Facies generator

Система должна генерировать интервалы:

```text
top
base
facies
lithology
trend
```

Поддержать минимум:

```text
shale
clean_sandstone
shaly_sandstone
tight_sandstone
limestone
dolomite
```

### FR-3. Hidden truth

Система должна генерировать по depth grid:

```text
Vsh
PHI
Sw
fluid
lithology
facies
is_reservoir
is_pay
bad_hole_mask
```

### FR-4. Forward log model

Система должна генерировать кривые:

```text
GR
RHOB
NPHI
DT
RT
CALI
```

### FR-5. Realism layer

Система должна иметь интерфейс:

```python
class IRealismEnhancer:
    def enhance(self, base_curves, truth, scenario, rng):
        ...
```

Минимально реализовать:

```text
NoOpRealismEnhancer
StatisticalRealismEnhancer
```

Подготовить место для:

```text
AutoencoderMCMCRealismEnhancer
DiffusionResidualEnhancer
```

### FR-6. Physics constraints

После realism layer система должна проверять и исправлять явные физические нарушения:

```text
curve range checks
facies-specific bounds
RHOB/PHI consistency
GR/Vsh consistency
RT/Sw consistency
CALI/bad hole consistency
```

### FR-7. Artifact simulator

Артефакты должны быть явно отражены в ground truth.

### FR-8. Export

Система должна экспортировать:

```text
well.las
truth.json
manifest.json
preview.html or preview.png
```

## 5. Нефункциональные требования

- Deterministic generation by seed.
- Core generation for one 300m well with 0.1m step should be fast enough for interactive use.
- Core should run without GPU.
- ML modules optional.
- All generated LAS files must be readable by `lasio.read()`.
- Unit tests required.

## 6. Acceptance criteria

1. CLI generates output files.
2. LAS contains correct curves and metadata.
3. `truth.json` contains interval-level and curve-level hidden truth.
4. Generated curves have plausible ranges.
5. Same seed gives same result.
6. At least two scenarios are implemented.
7. Tests cover depth grid, facies sequence, curve generation, LAS export, ground truth export, artifact masks.
