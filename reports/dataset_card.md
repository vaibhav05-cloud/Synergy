# DeploySense Dataset Card

## Purpose
This dataset supports a pre-deployment classifier that assigns **Low**, **Medium**, or **High** risk to a candidate release. One row is one completed GitHub Actions workflow run that qualifies as an explicit deployment workflow or a mainline delivery proxy.

## Sources
- Public GitHub REST API workflow-run metadata.
- Public GitHub REST API commit metadata and changed file paths.
- Causally generated operational context for facts GitHub does not publish: on-call availability, post-deployment incidents, and rollbacks.

## Leakage policy
The model dataset excludes the current run conclusion, completion timestamp, duration, synthetic outcome columns, risk score, calibrated risk index, and latent reliability shock. Historical failure and incident features contain only records strictly earlier than the candidate timestamp.

## Target policy
`risk_level` is derived from a calibrated pre-deployment risk index with thresholds calculated for this collection: Low < 45.97, Medium < 79.39, High otherwise. The index blends global, repository-relative, and domain-relative risk so the target is less dominated by one unusually busy repository.

## Dataset snapshot
- Rows: 1482
- Public repositories: 12
- Time range: 2025-12-18T00:25:52+00:00 to 2026-07-14T11:17:37+00:00
- Split strategy: stratified_temporal; 70% train, 15% validation, 15% test within each risk class when stratified.

## Appropriate use
Use `data/processed/model_features.csv` for model training. Use `deployment_risk_master.csv` only for auditing and explanation work; it contains post-deployment outcomes and must not be passed directly to a model.

## Limitations
GitHub Actions workflow runs are a public proxy for deployments, not an organization's private production-deployment record. Operational fields are explicitly synthetic and intended for a hackathon simulation, not for real production approval without organizational incident and paging data.
