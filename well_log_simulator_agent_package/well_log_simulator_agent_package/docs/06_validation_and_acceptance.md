# Validation, testing and acceptance criteria

## 1. Виды валидации

```text
1. Structural validation
2. Physical validation
3. Statistical validation
4. Educational validation
5. Export validation
```

## 2. Structural validation

Проверить:

```text
- depth monotonicity;
- no unexpected NaN except configured gaps;
- all requested curves exist;
- curve lengths match depth length;
- intervals cover required range;
- ground truth intervals do not overlap incorrectly.
```

## 3. Physical validation

- GR/Vsh: higher Vsh should generally map to higher GR.
- RHOB/PHI: higher porosity should generally lower RHOB for same matrix/fluid.
- NPHI: should broadly follow porosity, with shale and gas corrections.
- DT: higher porosity / shale should generally increase DT.
- RT/Sw: lower water saturation should generally increase resistivity.
- CALI: bad-hole/washout intervals should show increased CALI.

## 4. Statistical validation

Compare synthetic vs real/calibration dataset:

```text
- per-curve distributions;
- facies-specific distributions;
- correlation matrix;
- autocorrelation / vertical continuity;
- crossplots;
- PCA/t-SNE/UMAP qualitative view.
```

## 5. Educational validation

A generated case is valid if:

```text
- target learning objective is present;
- answer is not ambiguous beyond tolerance;
- ground truth can be used for automatic checking;
- student-visible curves do not leak hidden truth;
- artifacts are understandable and intentional.
```

## 6. Export validation

Every generated LAS must pass:

```python
las = lasio.read(path)
assert "DEPT" in las.keys()
```

And must include well metadata, curve metadata, units, NULL value, expected mnemonics.

## 7. Acceptance tests

### Test 1: deterministic generation

Same config + same seed → identical output curves and truth.

### Test 2: gas sand scenario

Expected: at least one clean sandstone pay interval, high RT, RHOB/NPHI gas effect, ground truth fluid = gas.

### Test 3: washout scenario

Expected: CALI increases in bad-hole interval, bad_hole_mask true, RHOB/NPHI reliability flags reduced.

### Test 4: LAS roundtrip

Generated LAS can be read by lasio.

### Test 5: constraints

Constraint violation rate below threshold.
