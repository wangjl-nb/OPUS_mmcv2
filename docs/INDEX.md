# Project Docs Index

This folder centralizes all markdown documentation for fast project onboarding and session handoff.

## Recommended Read Order (for new sessions)

1. `TASK_WORKORDER_CURRENT.md`
2. `PROJECT_GUIDE.md`
3. `LIDAR_FEATURE_EXTRACTOR_FLOW.md`
4. `INSTALL_opus_gpt.md`
5. `README.md`

## Core Project Docs

- `README.md`
  Main project intro, model zoo, training/eval usage.
- `PROJECT_GUIDE.md`
  Project architecture, code-entry mapping, troubleshooting knowledge.
- `LIDAR_FEATURE_EXTRACTOR_FLOW.md`
  Detailed LiDAR branch tensor flow and shape derivation.
- `INSTALL_opus_gpt.md`
  Environment setup and validation commands.
- `TASK_WORKORDER_TEMPLATE.md`
  Blank workorder template.
- `TASK_WORKORDER_CURRENT.md`
  Current task context and handoff log.

## Dataset Template Docs

- `templates/new_dataset/README.md`
- `templates/new_dataset/info_schema.md`

## Third-Party Docs (MapAnything mirror)

- `third_party/map-anything/README.md`
- `third_party/map-anything/train.md`
- `third_party/map-anything/CHANGELOG.md`
- `third_party/map-anything/CONTRIBUTING.md`
- `third_party/map-anything/CODE_OF_CONDUCT.md`
- `third_party/map-anything/data_processing/README.md`
- `third_party/map-anything/data_processing/wai_processing/download_scripts/README.md`
- `third_party/map-anything/benchmarking/calibration/README.md`
- `third_party/map-anything/benchmarking/dense_n_view/README.md`
- `third_party/map-anything/benchmarking/rmvd_mvs_benchmark/README.md`

## Notes

- Markdown files were moved into `docs/` with path mirroring (e.g. `templates/...` -> `docs/templates/...`).
- Source code, configs, and assets stay in their original directories.
- For path-sensitive commands, run from repo root unless a doc explicitly says otherwise.
