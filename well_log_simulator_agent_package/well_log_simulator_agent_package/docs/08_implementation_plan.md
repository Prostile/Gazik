# Implementation plan

## Milestone 1 — Core skeleton

- package structure;
- Pydantic ScenarioConfig;
- DepthGrid;
- GeneratedWell model;
- CLI generate command;
- basic tests.

## Milestone 2 — Facies and truth

- SemiMarkovFaciesGenerator;
- facies intervals;
- PetrophysicalTruthGenerator;
- interval-level ground truth;
- curve-level truth arrays.

## Milestone 3 — Forward logs

- GR;
- RHOB;
- NPHI;
- DT;
- RT;
- CALI;
- base curves dataframe.

## Milestone 4 — Artifacts

- noise;
- spikes;
- washout;
- gaps;
- depth shift;
- artifact metadata in truth.

## Milestone 5 — Export

- LAS export via lasio;
- truth JSON;
- manifest JSON;
- preview plot.

## Milestone 6 — Data ingestion

- LAS importer;
- curve aliases;
- QC;
- resampling;
- Parquet export.

## Milestone 7 — Realism layer v1

- NoOpRealismEnhancer;
- StatisticalRealismEnhancer;
- constraints after enhancement.

## Milestone 8 — Autoencoder/MCMC prototype

- training windows;
- small Conv1D autoencoder;
- latent sampling stub;
- residual blending;
- metrics report.

## Milestone 9 — Future v3 preparation

- DiffusionResidualEnhancer interface stub;
- TSGANImputer interface stub;
- AnswerEvaluator interface stub.
