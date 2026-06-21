# Risks and open questions

## Technical risks

### 1. Unrealistic physics-only curves

Mitigation: start with educational correctness; add statistical realism; calibrate against real LAS.

### 2. ML realism layer breaks hidden truth

Mitigation: use residual blending; apply physics constraints after ML; measure constraint violation rate.

### 3. Real LAS are heterogeneous

Mitigation: curve aliases; unit normalization; QC flags; do not require all curves for every well.

### 4. Lack of labeled facies

Mitigation: start with synthetic labels; use electrofacies clustering; allow manual calibration.

### 5. Overengineering with ML

Mitigation: NoOpRealismEnhancer must always work; StatisticalRealismEnhancer before Autoencoder/MCMC; Diffusion only after dataset and metrics exist.

## Domain questions for expert review

- What facies set should be supported first?
- Which lithologies are required for university exercises?
- Which curves are mandatory in local curriculum?
- Should saturation be calculated with Archie only at first?
- What tolerance should be used for top/base checking?
- Should units be meters only or meters/feet?
- Should LAS 2.0 be the only export target?
