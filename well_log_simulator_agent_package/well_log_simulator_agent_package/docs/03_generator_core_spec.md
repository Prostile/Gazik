# Спецификация ядра генератора

## 1. Основной pipeline

```text
ScenarioConfig → DepthGrid → FaciesSequence → PetrophysicalTruth → BaseCurves → RealismEnhancer → PhysicsConstraints → ArtifactSimulator → Exports
```

## 2. DepthGrid

Вход: `start`, `stop`, `step`, `unit`.

Требования:

- Поддержать meters.
- Проверять, что `step > 0`.
- Гарантировать монотонный depth.
- Использовать rounding policy, чтобы избежать накопления float-ошибок.

## 3. FaciesSequence

Использовать semi-Markov generator:

1. Выбрать текущую facies.
2. Сэмплировать толщину пласта из distribution.
3. Заполнить интервал.
4. Выбрать следующую facies по transition matrix.
5. Повторять до `depth.stop`.

Минимальные facies:

```text
shale
clean_sandstone
shaly_sandstone
tight_sandstone
limestone
dolomite
```

Пример transition matrix для clastic setting:

```yaml
transition_matrix:
  shale:
    shale: 0.55
    shaly_sandstone: 0.30
    clean_sandstone: 0.15
  shaly_sandstone:
    shale: 0.30
    shaly_sandstone: 0.45
    clean_sandstone: 0.20
    tight_sandstone: 0.05
  clean_sandstone:
    shale: 0.15
    shaly_sandstone: 0.25
    clean_sandstone: 0.50
    tight_sandstone: 0.10
```

## 4. PetrophysicalTruthGenerator

Для каждого depth sample генерировать:

```text
facies, lithology, Vsh, PHI, Sw, fluid, is_reservoir, is_pay
```

### Facies-specific priors

| Facies | Vsh | PHI | Sw |
|---|---:|---:|---:|
| shale | 0.65–1.00 | 0.03–0.15 | 0.80–1.00 |
| shaly_sandstone | 0.25–0.60 | 0.08–0.22 | 0.35–1.00 |
| clean_sandstone | 0.00–0.20 | 0.15–0.32 | 0.20–1.00 |
| tight_sandstone | 0.00–0.25 | 0.03–0.12 | 0.60–1.00 |
| limestone | 0.00–0.15 | 0.02–0.25 | 0.25–1.00 |
| dolomite | 0.00–0.15 | 0.02–0.20 | 0.25–1.00 |

### Correlations

Не генерировать Vsh/PHI/Sw полностью независимо. Нужны корреляции:

```text
higher Vsh → higher GR
higher PHI → lower RHOB
higher PHI → higher NPHI
higher PHI → higher DT
lower Sw with hydrocarbon → higher RT
gas → RHOB/NPHI crossover effect
```

На старте можно реализовать correlated sampling через Gaussian copula или multivariate normal in transformed space.

## 5. ForwardLogModel

### GR

```text
GR = GR_clean + Vsh * (GR_shale - GR_clean) + noise
```

### RHOB

```text
RHOB = (1 - PHI) * rho_matrix + PHI * rho_fluid
```

### NPHI

```text
NPHI = PHI + shale_bound_water_effect + lithology_offset + gas_effect + noise
```

### DT

```text
DT = PHI * DT_fluid + (1 - PHI) * DT_matrix + shale_effect + compaction_effect + noise
```

### RT

Clean formation стартовая модель:

```text
Rt = a * Rw / (PHI^m * Sw^n)
```

Типично: `a = 1`, `m = 2`, `n = 2`.

Для shaly sands позже добавить Simandoux / Indonesia / Waxman-Smits.

### CALI

```text
CALI = bit_size + washout_effect + rugosity_noise
```

## 6. ArtifactSimulator

Артефакты должны быть независимым слоем после базовой генерации.

- Noise: Gaussian white noise + correlated noise.
- Spikes: random spike count + curve-specific allowed spikes.
- Missing intervals: replace curve intervals by NULL value.
- Washout: CALI increases, RHOB/NPHI degraded, bad_hole_mask = true.
- Depth shift: shift selected curve by 0.2–2.0 m.

## 7. Output objects

```python
class GeneratedWell:
    well_id: str
    depth: np.ndarray
    curves: pd.DataFrame
    truth: GroundTruth
    scenario: ScenarioConfig
    manifest: GeneratedWellManifest
```
