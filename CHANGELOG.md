# Changelog

## Unreleased — Educational suite

- Added siltstone, marl, coal and anhydrite with shared petrophysical properties,
  curve constraints and reservoir/pay rules.
- Added shale-corrected resistivity configuration and five educational scenarios.
- Added student-answer scoring for reservoir, pay, lithology and recording quality.
- Added raw LAS collection conventions, example data and structure validation CLI.
- Unified forward/scoring expected-curve physics, added guaranteed required intervals
  and separated facies scoring from generalized lithology scoring.

## 0.3.0 — Full Hybrid #2

- Added strict five-channel calibration contract and electrofacies window labels.
- Added latent statistics per condition and condition-aware proposal selection.
- Replaced independent rejection attempts with Metropolis-like constrained search.
- Unified runtime candidate scoring and final physics-constraint reporting.
- Added statistical pass/warning/fail gates and strict CLI behavior.
- Expanded realism manifest reports and quality tests.

## 0.2.0 — Hybrid #2 MVP

- Added LAS ingestion, calibration datasets, Conv1D autoencoder artifacts and
  residual-only realism generation.
