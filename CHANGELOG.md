# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as published via `project(VERSION ...)` in CMake and `./run_mono --version`.

## [Unreleased]

- Ongoing work toward broader product readiness (see `plan.md`).
- TUM runner: optional `--tum-camera-config <calib.json>` to override built-in freiburg1 intrinsics (see `config/examples/tum_pinhole_fr1.json`).
- Operations hooks: `Tracking::runStatistics()` (reloc / lost-frame / re-init counters), `--run-summary-json <path>` (single-line JSON), `--strict-exit` (exit code 3 if tracking did not finish in `OK`).
- Research benchmark matrix: `eval/leaderboard_suite.json` plus `scripts/build_leaderboard.py` (TUM windows × methods, mean ATE, mean rank; not a KITTI-leaderboard clone); shared `scripts/eval_lib.py` with `check_regression_gate.py`.
- Academic reuse: `CITATION.cff` and a **Citing** section in `README.md` (BibTeX + reproducibility note).
- CLI: `run_mono --help` / `-h` (full TUM flags); fix vocab path selection when `--strict-exit` is present; `build_leaderboard.py --dry-run` (planned matrix only). CI runs `run_mono --help` and leaderboard dry-run.
- Evaluation: `eval/comparison_protocol.md` — fair ATE comparison vs other OSS (stella_vslam as primary BSD peer; match modality and `evo_ape` alignment).
- Comparison verification: `scripts/verify_comparison_benchmark.sh` (preset TUM runs + mean ATE) and `scripts/print_ate_mean.py` (uses `regression_baselines.json` evo flags).

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
