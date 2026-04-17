# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as published via `project(VERSION ...)` in CMake and `./run_mono --version`.

## [Unreleased]

### Added

- `--accel` now drives a working gravity prior on mono runs. Mono init applies accelerometer-based world-frame alignment (previously only RGB-D init did), so `BA: Added N gravity prior residuals` actually fires in monocular BA. Opt-in measurements on head-250 (`evo_ape --align --correct_scale --t_max_diff 0.05`): `fr1_xyz` mono `0.028136 → 0.024931` (-11.4%), `fr1_room` mono `0.176506 → 0.164001` (-7.1%).
- Two new regression gates to lock in the above: `room_mono_accel_head_repro` (ceiling `0.170 m`) and `xyz_mono_accel_head_repro` (ceiling `0.027 m`).
- Post-relocalization emergency-KF cooldown: after a successful `relocalize()`, the low-tracked-features emergency KF insertion is deferred for `kPostRelocEmergencyKfCooldownFrames = 3` frames to avoid post-recovery KF bursts. `room_mono_head_repro` 0.197 → 0.177 (-10.5%); `xyz_mono_head_repro` unchanged.
- `Frame::depth_is_learned_` / `Keyframe::depth_is_learned_` flag plus a separate BA sigma for learned metric depth (`0.15 m` vs `0.015 m` for sensor metric depth). Prevents ONNX metric-depth outputs from being trusted at millimeter level in BA.

### Changed

- `Landmark::is_bad_` is now `std::atomic<bool>` (was plain `bool`). Fixes a ThreadSanitizer-flagged data race between `Optimizer::bundleAdjustment -> Landmark::setBad()` on the LocalMapping thread and `Landmark::isBad()` hot-path reads on the Tracking thread. No regression-gate impact; closes one race class.
- `Tracking::addFrame` now holds `pose_mutex_` across the `current_frame_` / `last_frame_` shared_ptr swap. `onBACompleted` already takes the same mutex on the read side; the one-sided lock was racing under TSan. Closes a second race class.
- `Tracking::trackLocalMap` snapshots `kf->landmarks_` under `Keyframe::mutex_` before iterating, and `LocalMapping::createNewMapPoints` takes each keyframe's `mutex_` individually when writing `kf->landmarks_[idx]`. Closes the third race class. Combined with the above two, 3-rep 600-frame async `fr1_room` mono no longer core-dumps (was 1/3 crash rate before the three race fixes).
- 600-frame loop-enabled `room_depth` is materially more stable across runs: median `0.617 → 0.126 m`, worst-case `0.871 → 0.146 m` (three reps at HEAD, loop closing on, same command as before). Credited to the post-reloc cooldown and the simplification pass that landed in the same branch.
- `eval/stella_comparison_results.md` / `eval/stella_comparison.json` refreshed: `room_mono` narrowed from `0.22050 → 0.17651 m` (~8x → ~6.4x of stella head-250), other gates byte-identical vs the 2026-04-15 snapshot.

### Removed

- Experimental `ScoreReferenceKeyframePolicy` and `PipelineReferenceKeyframePolicy` and their surrounding infrastructure (`src/experiments/reference_keyframe/`, `tools/reference_policy_experiments.cc`, `scripts/eval_reference_policies.sh`, `scripts/update_reference_policy_docs.py`, `docs/*.md`, `experiments/reference_keyframe/`). Only `HeuristicReferenceKeyframePolicy` remains; the `ReferenceKeyframePolicy` contract in `src/core/` is kept in case a future experiment plugs back in. Net: ~2000 lines deleted with byte-identical trajectory SHAs on all gates.
- `--keyframe-trace-csv` CLI flag, `Tracking::setKeyframeDecisionTraceSink`, `Tracking::traceKeyframeDecision`, and their nine call sites inside `needNewKeyframe()`. No external consumer read the CSV; removal is behavior-neutral (the flag early-returned when unset).
- Various trace-only stdout lines in `tracking.cc`: `RoomFocusTrace` (freiburg1_room-specific frame-range hack), `LocalMapVisibility` bucket dump, `FallbackCandidateTrace` top-N candidate table, `BootstrapStats` aggregate dump, and Phase B `FallbackInlier` / `FallbackSummary` plumbing. Their findings are preserved in `eval/room_mono_frame199_diag.txt`; removing the live prints drops ~230 lines without behavior change.
- Dead locals and orphaned struct fields behind the removed traces (`fallback_two_nn` / `fallback_reject_*` / `MatchCandidate::ratio_margin` / `MatchCandidate::coarse_err_px` / most fields of `PoseFilterStats` and `LocalMapSourceStats`, plus the out-param on `bootstrap_coarse_ok_and_err`).
- V1 coarse-reprojection-error tiebreak in the late sparse mono fallback sort. `plan.md` §8.3 step 7 had documented it as near-noise (`0.197374 → 0.197177`); empirically confirmed after removal (drift `0.176340 → 0.176506`, within run-to-run noise).

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
