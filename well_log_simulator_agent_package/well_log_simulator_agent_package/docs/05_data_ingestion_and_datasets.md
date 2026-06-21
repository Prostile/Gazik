# Data ingestion и датасеты

## 1. Цель

Создать pipeline для загрузки реальных LAS и калибровки генератора/realism layer.

## 2. Основные источники данных

### NLOG

NLOG — основной кандидат для реальных LAS из Нидерландов. Он содержит borehole-level data, включая digital log curves в LIS/LAS после истечения confidentiality period.

Использование:

```text
- калибровка распределений;
- curve mnemonic aliases;
- реальные missing intervals;
- статистика шумов;
- training windows для realism layer.
```

### Volve

Volve полезен для будущих задач, где well logs связаны с drilling data.

### Netherlands F3 / seismic datasets

Полезно позже для seismic extension: seismic tie, synthetic seismogram, future inversion exercises.

## 3. Ingestion pipeline

```text
raw LAS files
  ↓
lasio read
  ↓
metadata extraction
  ↓
curve mnemonic canonicalization
  ↓
unit normalization
  ↓
depth resampling
  ↓
QC masks
  ↓
window segmentation
  ↓
Parquet dataset
```

## 4. Canonical curve names

```yaml
DEPT:
  aliases: [DEPT, DEPTH, MD]
GR:
  aliases: [GR, GRC, GAM, CGR]
RHOB:
  aliases: [RHOB, RHOZ, DEN, ZDEN]
NPHI:
  aliases: [NPHI, TNPH, NPOR, NEU]
DT:
  aliases: [DT, DTC, AC]
RT:
  aliases: [RT, ILD, LLD, RDEP, RESD]
```

## 5. QC rules

```text
GR: 0 <= GR <= 250 API
RHOB: 1.5 <= RHOB <= 3.1 g/cc
NPHI: -0.15 <= NPHI <= 0.8 v/v
DT: 40 <= DT <= 200 us/ft or corresponding unit-converted range
RT: 0.1 <= RT <= 10000 ohm.m
```

## 6. Output Parquet schema

```text
well_id: string
depth: float
curve: string
value: float
canonical_curve: string
unit: string
source_file: string
qc_flag: string
```

Windowed dataset:

```text
window_id
well_id
top
base
curves_tensor_path
available_curves
electrofacies_label optional
```

## 7. Dataset governance

- Не коммитить большие LAS в основной git repo.
- Держать raw data отдельно.
- Generated/intermediate data хранить в Parquet.
- Для ML использовать manifest files.
- Сохранять provenance: source, license/access note, preprocessing version, curve aliases version.
