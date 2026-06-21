# Research papers digest for implementation

This document is an engineering reading guide for the AI agent implementing the simulator.

## 1. Fonseca & Saraiva, 2026

**Title:** Synthetic Well Log Generation with Preserved Multivariate Correlations and Vertical Facies Stacking Patterns  
**URL:** https://arxiv.org/abs/2605.06255  
**Core idea:** Markov chain models + autoencoder dimensionality reduction + MCMC sampling in latent space.  
**Generated/handled data:** Density, P-Sonic, S-Sonic; electrofacies stacking; rock-physics relationships.

### Why it matters

This is the closest paper to the current target architecture. It explicitly combines:

```text
vertical facies stacking
multivariate correlations
latent-space sampling
geological realism
```

### What to use

- Use Markov/semi-Markov facies sequence as structural backbone.
- Use autoencoder latent representation for multivariate windows.
- Use MCMC-style sampling for realistic combinations.
- Preserve physics constraints after sampling.

### What not to copy blindly

- The paper focuses on elastic logs and seismic scenario testing, not on educational LAS with GR/RHOB/NPHI/DT/RT/CALI.
- Our implementation must preserve ground truth for automatic checking.

## 2. Al-Fakih et al., 2024

**Title:** Well log data generation and imputation using sequence-based generative adversarial networks  
**URL:** https://arxiv.org/abs/2412.00718  
**Core idea:** TSGAN for synthetic well log generation; SeqGAN for imputation.  
**Curves:** GR, DT, NPHI, ILD, RHOB.

### Why it matters

Useful for future Hybrid #3:

```text
generation of sequence-like well logs
imputation of missing intervals
comparison metrics for generated logs
```

### What to use

- Preprocessing pattern: cleaning, normalization, segmentation, model training, PCA/t-SNE comparison.
- Future imputation tasks: create gaps, recover, compare to hidden truth.

### What not to use for core

Do not use GAN as the primary generator of educational scenarios. It is less controllable and does not naturally produce pedagogical ground truth.

## 3. Miele & Linde, 2025

**Title:** Diffusion models for multivariate subsurface generation and efficient probabilistic inversion  
**URL:** https://arxiv.org/abs/2507.15809  
**Core idea:** diffusion models for multivariate subsurface modeling, conditioning by hard well logs and seismic.

### Why it matters

Important for future advanced mode:

```text
conditional generation
uncertainty
probabilistic inversion
multivariate subsurface properties
```

### What to use later

- Diffusion as residual/variation generator.
- Conditioning mechanism idea.
- Use well logs as hard conditions.

## 4. Chen et al., 2026

**Title:** Multi-Condition Guided Diffusion Model for Controllable Elastic Parameter Synthesis  
**URL:** https://arxiv.org/abs/2606.05751  
**Core idea:** multi-condition guided diffusion for controllable Vp/Vs/density synthesis.

### Why it matters

Shows how to guide generation with multiple condition types: implicit constraints, structural constraints, explicit conditioning operators.

### What to use later

- Design future `DiffusionResidualEnhancer` around explicit conditioning.
- Keep scenario config compatible with future conditioning fields.

## 5. Lopes & Jorge, 2017

**Title:** Mind the Gap: A Well Log Data Analysis  
**URL:** https://arxiv.org/abs/1705.03669  
**Core idea:** gap statistics and missing interval prediction for neutron porosity logs.

### Why it matters

Supports educational QC/imputation exercises.

### What to use

- Generate artificial gaps.
- Track gap metadata in ground truth.
- Build missing-log recovery tasks.

## 6. Kanfar et al., 2020

**Title:** Real-Time Well Log Prediction From Drilling Data Using Deep Learning  
**URL:** https://arxiv.org/abs/2001.10156  
**Core idea:** Inception CNN + Temporal Convolutional Network for predicting density/porosity/sonic logs from drilling data.

### Why it matters

Useful later if the product includes drilling-data-to-log exercises.

## 7. Azevedo et al., 2018

**Title:** Geostatistical Rock Physics AVA Inversion  
**URL:** https://arxiv.org/abs/1810.06552  
**Core idea:** stochastic sequential simulation + facies-dependent rock physics + seismic/AVA inversion.

### Why it matters

Important long-term reference for professional mode: facies-dependent rock physics, uncertainty-aware simulations, future seismic extension.

## 8. WLFM, 2025

**Title:** WLFM: A Well-Logs Foundation Model for Multi-Task and Cross-Well Geological Interpretation  
**URL:** https://arxiv.org/abs/2509.18152  
**Core idea:** foundation model pretrained on multi-curve logs from 1200 wells; tokenization of log patches; masked-token modeling; stratigraphy-aware contrastive learning.

### Why it matters

Future AI evaluator / assistant: masked curve reconstruction, lithology classification, porosity estimation, student-answer feedback.

## Engineering conclusion

Implement in this order:

```text
1. Semi-Markov + physics generator
2. Data ingestion/calibration
3. Statistical realism
4. Autoencoder/MCMC residual realism
5. GAN/diffusion residuals
6. Foundation-model evaluator
```
