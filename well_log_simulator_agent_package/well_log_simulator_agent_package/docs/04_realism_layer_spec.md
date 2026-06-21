# Спецификация realism layer: Autoencoder/MCMC + constraints

## 1. Назначение

Realism layer повышает похожесть синтетических кривых на реальные well logs, но не управляет геологической истиной.

```text
Truth first. Realism second. Constraints always.
```

## 2. Интерфейс

```python
class IRealismEnhancer(Protocol):
    def enhance(
        self,
        base_curves: pd.DataFrame,
        truth: GroundTruth,
        scenario: ScenarioConfig,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        ...
```

## 3. Реализации

### NoOpRealismEnhancer

Возвращает `base_curves` без изменений. Нужен для тестов и MVP.

### StatisticalRealismEnhancer

Добавляет:

```text
- correlated residual texture;
- facies-specific curve variability;
- distribution matching;
- small curve-to-curve covariance.
```

### AutoencoderMCMCRealismEnhancer

Целевая advanced-реализация для Hybrid #2.

## 4. Autoencoder/MCMC training pipeline

### Input data

Вход: набор реальных LAS.

Минимальный набор кривых:

```text
GR
RHOB
NPHI
DT
RT/ILD
```

### Canonicalization

```text
Gamma Ray: GR, GRC, GAM, CGR
Bulk density: RHOB, RHOZ, DEN, ZDEN
Neutron porosity: NPHI, TNPH, NPOR
Sonic: DT, DTC, AC
Resistivity: RT, ILD, LLD, RDEP, RESD
```

### QC

- Remove impossible values.
- Convert units if needed.
- Resample to common depth step.
- Mask missing intervals.
- Normalize per curve.
- Segment into windows.

### Window representation

```text
window_size: 64 / 128 / 256 samples
channels: GR, RHOB, NPHI, DT, RT
shape: [batch, channels, window_size]
```

### Autoencoder

Стартовая архитектура:

```text
1D Conv Encoder → latent vector z → 1D Conv Decoder
```

Loss:

```text
MSE reconstruction + optional derivative/smoothness loss + optional correlation loss
```

### Latent sampling

Варианты:

1. Fit Gaussian/GMM/KDE per electrofacies/cluster and sample z.
2. MCMC: sample latent z under constraints; reject/penalize decoded windows that violate physics constraints.
3. Conditional: condition by facies/electrofacies, depth trend, reservoir type.

## 5. Как не сломать ground truth

Предпочтительный вариант:

```text
base_curves = physics_forward_model(truth)
decoded_realistic_window = autoencoder_decoder(z)
residual_texture = decoded_realistic_window - smoothed(decoded_realistic_window)
final_curves = base_curves + alpha * residual_texture
```

Не делать так:

```text
final_curves = decoded_realistic_window
```

## 6. Physics constraint layer

После enhance выполнить:

```text
- clip ranges;
- validate facies-specific bounds;
- check RHOB-PHI relation;
- check RT-Sw consistency;
- preserve gas effect where required;
- preserve washout effects;
- preserve interval-level interpretation.
```

## 7. Metrics

- Per-curve histograms.
- Correlation matrix.
- Crossplots: RHOB vs NPHI, GR vs RT, DT vs RHOB.
- Vertical variogram/autocorrelation.
- Bed thickness distributions.
- PCA/t-SNE/UMAP qualitative comparison.
- Constraint violation rate.
