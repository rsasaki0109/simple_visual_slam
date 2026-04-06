# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as published via `project(VERSION ...)` in CMake and `./run_mono --version`.

## [Unreleased]

- Ongoing work toward broader product readiness (see `plan.md`).

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
