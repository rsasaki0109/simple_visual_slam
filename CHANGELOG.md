# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as published via `project(VERSION ...)` in CMake and `./run_mono --version`.

## [Unreleased]

- Nothing yet.

## [0.2.0] — 2026-04-14

### Added

- TUM runner: optional `--tum-camera-config <calib.json>` to override built-in freiburg1 intrinsics (see `config/examples/tum_pinhole_fr1.json`).
- Operations hooks: `Tracking::runStatistics()` (relocalization / lost-frame / re-init counters), `--run-summary-json <path>` (single-line JSON), and `--strict-exit` (exit code 3 if tracking did not finish in `OK`).
- Research benchmark matrix: `eval/leaderboard_suite.json` plus `scripts/build_leaderboard.py` (TUM windows x methods, mean ATE, mean rank) with shared `scripts/eval_lib.py`.
- Academic reuse: `CITATION.cff` and a **Citing** section in `README.md` (BibTeX + reproducibility note).
- Metric depth estimator support via `--metric-depth-model`.
- EuRoC stereo scaffolding via `--euroc-camera-config <calib.json>` and `--stereo`.
- CI smoke regression with deterministic synthetic test data (`scripts/generate_ci_test_data.py`, `scripts/download_ci_test_data.sh`, `scripts/run_ci_smoke_regression.py`).
- Fair `stella_vslam` comparison results in `eval/stella_comparison_results.md`.

### Changed

- Refactored tracking, loop closing, and optimizer internals with helper extraction for readability and maintenance.
- Mono initialization improved with median-parallax validation.
- Loop closing stabilized with cooldown `200`, relocalization locality checks, and overlap decay.
- Pose-graph construction now adds covisibility-weighted edges.
- Local bundle adjustment iterations increased from `10` to `15`.
- CLI: `run_mono --help` / `-h` now documents the full TUM flag set; vocab path selection is fixed when `--strict-exit` is present; `build_leaderboard.py --dry-run` prints the planned matrix only.
- Evaluation workflow: `eval/comparison_protocol.md` documents fair ATE comparison against other OSS, and `scripts/verify_comparison_benchmark.sh` / `scripts/print_ate_mean.py` verify published baselines with the same `evo_ape` settings as `eval/regression_baselines.json`.
- Regression baselines were tightened.

### Testing

- Test coverage expanded from `26` to `53` tests in the default build, or `55` with `-DUSE_DEPTH_DL=ON`.

## [0.1.0] — 2026-04

### Added

- Core SLAM stack: ORB tracking, local mapping, Ceres BA, optional DBoW2 loop closing.
- TUM RGB-D / EuRoC dataset runners; optional DL depth (ONNX); accelerometer priors.
- Reference-keyframe policy seam and evaluation harness (`scripts/eval_reference_policies.sh`).
- `--repro-eval` deterministic replay path; local regression gates (`scripts/check_regression_gate.py`, `eval/regression_baselines.json`).
- GitHub Actions CI (Ninja build + `ctest`); BSD-2-Clause `LICENSE`; `CONTRIBUTING.md`.
- `run_mono --version` reporting CMake `PROJECT_VERSION`.

### Notes

- Public API and ABI stability are not promised for 0.x; semver bumps track releases and breaking CMake/interface changes as they arise.
