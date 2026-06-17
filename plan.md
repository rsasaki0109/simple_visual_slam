# SimpleVisualSLAM Agent Handoff

This file is the authoritative handoff for Codex / Claude / Cursor as of **2026-04-21**.

Major update on this revision: **VIO Stage 0b + 0c landed.** A loosely-coupled visual-inertial pipeline (IMU preintegration in local BA, per-KF velocity + accel/gyro bias BA parameters, Forster 9-DoF preintegration residual, linear Visual-Inertial Initialization) is live on `master` and empirically validated on five EuRoC mono `--accel` sequences. See §16 for the VIO-specific handoff.

Reconnaissance commands used for this revision:

- `git log --oneline -25`
- `git diff --stat 3f9bc71..HEAD` (span since the last merged PR)
- `ctest --test-dir build --output-on-failure` (77/77 PASS)
- `python3 scripts/check_regression_gate.py --build build --all-gates --quiet` (7/7 PASS)
- `cat eval/regression_baselines.json`
- `cat eval/stella_comparison_results.md`
- `cat eval/metric_depth_test_results.md`
- `cat eval/euroc_test_results.md`
- `wc -l apps/*.cc src/*/*.cc src/*/*.h` → 12 090 lines
- `ls tests/test_*.cc` → 18 files
- `./build/run_mono --version` / `--help`
- EuRoC mono + `--accel` 5-sequence ATE sweep: MH_01/02/03, V1_01, V2_01 (Sim3 evo_ape)

If this document and the code disagree, the code is correct and this file should be updated.

---

## 1. Vision

**Readable SLAM** remains the core goal:

- a compact, BSD-licensed Visual SLAM system
- C++17 end to end
- easy to read, modify, benchmark, and extend
- one repository covering monocular SLAM, RGB-D, stereo depth, learned depth, loop closing, and lightweight robotics integration

The old tagline was "6k lines of C++17". That was true earlier in the project, but it is no longer literally true. The current measured count for `apps/*.cc` plus the requested `src/**/*.cc` and `src/**/*.h` files is **9715 lines**. The right updated statement for v0.2.0 is:

> **SimpleVisualSLAM is still small enough for one AI agent to understand end to end, but now large enough to cover mono, RGB-D, stereo, learned depth, loop closing, map I/O, and ROS2.**

What the project is trying to become:

- **Accuracy/stability target:** reach `stella_vslam`-class behavior on TUM head-250 and room revisit scenarios
- **Differentiator:** keep the learned-depth path first-class instead of bolted on
- **Engineering constraint:** stay BSD-friendly and keep Ceres/Sophus/OpenCV as the main stack; do not take a GPL shortcut unless explicitly approved
- **Research constraint:** every claimed improvement must be tied to a reproducible command and a recorded number

---

## 2. Current State (2026-04-21)

### 2.1 Snapshot

| Item | Current value |
| --- | --- |
| `master` HEAD | `46a726d` |
| `master` HEAD subject | `plan.md: record VIO Stage 0b/0c progress and validated ATE numbers` |
| Open PR branch | `vio-integration` (`8e6f4e7`), **PR #3** on GitHub, not yet merged |
| Version | `SimpleVisualSLAM 0.2.0` |
| Build used for this snapshot | `build` (no suffix — the Ninja-style build dirs `build_codex*` are legacy) |
| Recent change volume | `22 files changed, 2 820 insertions(+), 65 deletions(-)` in `3f9bc71..HEAD` (the span that introduced the whole VIO pipeline) |
| Unit tests | **77 / 77** passed |
| `ctest` wall time | `~10 sec` |
| Core app/source LOC | **12 090** lines across `apps/*.cc` + `src/*/*.cc` + `src/*/*.h` |
| Test source files | **18** `tests/test_*.cc` files |
| TUM regression gates | **7 / 7** PASS with bitwise-reproducible trajectories |
| EuRoC mono `--accel` sweep (5 sequences, 2026-04-20) | average ATE **2.10 m → 1.60 m (−24 %)** vs visual-only |

### 2.2 Supported Feature Matrix

| Feature | Status | Evidence |
| --- | --- | --- |
| Monocular SLAM | Implemented | `run_mono --tum ...` and default video path |
| RGB-D SLAM | Implemented | `--depth` |
| Accelerometer prior (BA gravity) | Implemented | `--accel`; per-KF gravity prior, weight env-tunable via `SVSLAM_BA_GRAVITY_PRIOR_WEIGHT` |
| **Loosely-coupled VIO (Stage 0b + 0c)** | **Implemented** | IMU preintegration (`sensors/imu_preintegrator.h`), BA preintegration residual (`VelocityPreintegrationError`, 9-DoF) with accel/gyro bias blocks, Forster rotation residual, Visual-Inertial Initialization (`tracking/visual_inertial_initializer.{h,cc}`). Active whenever `--accel` is paired with an IMU dataset (EuRoC) |
| EuRoC mono + IMU loader | Implemented | `--euroc <sequence_dir> --accel`; loads `cam0 T_BS` multi-line YAML, parses `imu0/data.csv`, plumbs extrinsic into Tracking |
| EuRoC stereo depth | Implemented, known-degraded | `--euroc ... --stereo`; stereo-only ATE on MH_01 is ~2.77 m vs 3.44 m mono — rectification pipeline does not handle EuRoC's fisheye-ish distortion well. Separate from the VIO work, see §8 |
| Stereo tracking mode | Partial | stereo depth is computed from `cam0+cam1`, tracking still runs on `cam0` |
| Relative DL depth | Implemented | `--depth-model <model.onnx>` with `-DUSE_DEPTH_DL=ON` |
| Metric DL depth | Implemented | `--metric-depth-model <model.onnx>` with `-DUSE_DEPTH_DL=ON` |
| Loop closing | Implemented | enabled in async runs when ORB vocabulary exists |
| Deterministic replay mode | Implemented | `--repro-eval` disables loop closing and runs local mapping synchronously |
| Map persistence | Implemented | writes `map.bin`, `trajectory.txt`, `trajectory_online.txt`, `trajectory_keyframes.txt` |
| Run summary JSON | Implemented | `--run-summary-json <path>` |
| Strict failure exit | Implemented | `--strict-exit` |
| ROS2 Jazzy node | Implemented, basic | `ros2/src/slam_node.cc`; currently no loop-closing parity, no VIO hookup |

### 2.3 CLI Surface

`./build_codex/run_mono --help` currently exposes:

- `--version`, `--help`
- `--euroc <sequence_dir>`
- `--tum <sequence_dir>`
- `--euroc-camera-config <calib.json>`
- `--stereo`
- `--tum-camera-config <calib.json>`
- `--depth`
- `--accel`
- `--repro-eval`
- `--reference-policy <heuristic|score|pipeline>`
- `--skip-frames N`
- `--max-frames N`
- `--depth-model <model.onnx>`
- `--metric-depth-model <model.onnx>`
- `--no-viz`
- `--run-summary-json <path>`
- `--keyframe-trace-csv <path>`
- `--strict-exit`

The ORB vocabulary is the last positional argument, otherwise the app searches `data/ORBvoc.txt` and then `ORBvoc.txt`.

### 2.4 Regression Gates

Measured on **2026-04-21** (`python3 scripts/check_regression_gate.py --build $PWD/build --data-tum $PWD/data/tum --all-gates --quiet`).

| Gate | Mode | Mean ATE (m) | Ceiling (m) | Status |
| --- | --- | ---: | ---: | --- |
| `xyz_depth_head_repro` | RGB-D | `0.011042` | `0.016000` | PASS |
| `xyz_mono_head_repro` | Mono | `0.028136` | `0.030000` | PASS |
| `xyz_mono_accel_head_repro` | Mono + accel | `0.024931` | `0.027000` | PASS |
| `room_depth_head_repro` | RGB-D | `0.079914` | `0.165000` | PASS |
| `room_depth_accel_head_repro` | RGB-D + accel | `0.057702` | `0.145000` | PASS |
| `room_mono_head_repro` | Mono | `0.176506` | `0.340000` | PASS |
| `room_mono_accel_head_repro` | Mono + accel | `0.164001` | `0.170000` | PASS (thin margin) |

**7 / 7 gates passing** with bitwise-identical trajectories on two back-to-back runs per gate.

Notes:
- VIO's preintegration residual is dormant on TUM runs (no `has_velocity_`) so these gates measure the non-VIO front-end/back-end behaviour — VIO work did not regress any gate.
- `room_mono_accel_head_repro` has the smallest pass margin (6 mm). Any mono front-end change must re-run this gate before and after.
- `room_mono_head_repro` is still the widest gap to `stella_vslam` (§2.5 / §10.4).

### 2.5 `stella_vslam` Comparison Status

Important nuance:

- the **Fair Head-250** SimpleVisualSLAM rows in `eval/stella_comparison_results.md` were refreshed on **2026-04-15** (`2ac7ffa`, `scripts/verify_comparison_benchmark.sh`, `BUILD=build_codex`); **`stella_vslam` baselines** in that table are still the original fair-window numbers from `stella_eval`
- **lower sections** of the same markdown file (loop-enabled median runs, 600-frame triplets, covis-weight notes, etc.) are **historical snapshots** unless explicitly re-run
- machine-readable partner: `eval/stella_comparison.json` (updated for the fair table on the same date)

Fixed `stella_vslam` head-250 baselines from `eval/stella_comparison_results.md`:

- `xyz_depth`: `0.00889256 m`
- `xyz_mono`: `0.01413570 m`
- `room_depth`: `0.02110508 m`
- `room_mono`: `0.02743546 m`

Best-known / current SimpleVisualSLAM numbers to compare against them:

| Scenario | SimpleVisualSLAM number | Source | Gap vs `stella_vslam` |
| --- | ---: | --- | ---: |
| `xyz_depth` | `0.011042` | current repro gate | `1.24x` |
| `xyz_mono` | `0.036` best retained after `c5bcbe1` | retained project note | `2.55x` |
| `xyz_mono` | `0.028136` current repro gate | current gate | `1.99x` |
| `room_depth` | `0.0695` best retained after `8007ee2` | retained project note | `3.29x` |
| `room_depth` | `0.079914` current repro gate | current gate | `3.79x` |
| `room_mono` | `0.197374` current repro gate | current gate | `7.20x` |

Practical reading:

- `xyz_depth` is close
- `xyz_mono` is back under its strict repro gate, but still materially above `stella_vslam`
- `room_depth` is still substantially behind even after pose-graph improvements
- `room_mono` remains the hardest gap by far

### 2.6 Metric Depth Test Results

From `eval/metric_depth_test_results.md` and the metric-depth appendix in `eval/stella_comparison_results.md`:

- Build with DL enabled succeeded: `cmake -S . -B build_codex3 -G Ninja -DBUILD_TESTS=ON -DUSE_DEPTH_DL=ON && cmake --build build_codex3 -j$(nproc)`
- Working indoor metric model path: `models/depth_anything_v2_metric_indoor_small.onnx`
- Local model details:
  - size: `95M`
  - SHA256: `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c`
- Official/public Hugging Face indoor ONNX URLs attempted on 2026-04-14 returned `401 Unauthorized`

Measured results:

| Sequence | Frames | ATE (m) | Result |
| --- | ---: | ---: | --- |
| `freiburg1_xyz` | `50` | `0.01093245` | PASS, trajectory written |
| `freiburg1_xyz` | `250` | `0.06528716` | PASS, trajectory written |
| `freiburg1_room` | `50` | `0.01712747` | PASS, trajectory written |
| `freiburg1_room` | `250` | `0.38429766` | functional but much worse than sensor depth |

Interpretation:

- the metric-depth pipeline is **real and working**
- the dynamic-output-shape crash was fixed by `a32d904`
- `xyz` looks promising
- `room` at `250` frames is still poor, and loop candidates were rejected by `computeSim3()`

### 2.7 EuRoC Real-Dataset Verification (Updated 2026-04-20)

Real EuRoC sequences are now on disk under `datasets/euroc/` — downloaded via the **Wayback Machine** because the ETH ASL origin (`robotics.ethz.ch/~asl-datasets/...`) is unreachable from this environment. The working recipe:

```
curl -L --fail --retry 3 -o datasets/euroc/<SEQ>.zip \
  "https://web.archive.org/web/<TS>if_/http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/<path>/<SEQ>.zip"
```

The `if_` modifier on the Wayback URL returns the raw ZIP rather than the framed viewer. Find a valid `<TS>` with `curl -sSI "https://web.archive.org/web/2023/..."` and follow the 302 redirect. Typical capture size ≈ original (1–1.6 GB), download ≈ 4 min at ~6 MB/s.

Sequences available on disk:

| Sequence       | ZIP (GB) | Cam frames | IMU samples | GT poses | Ground truth file |
| ---            | ---:     | ---:       | ---:        | ---:     | ---               |
| `MH_01_easy`   | 1.57     | 3 683      | 36 820      | 36 382   | `datasets/euroc/MH_01_easy/gt_tum.txt` |
| `MH_02_easy`   | 1.29     | 3 041      | 30 400      | 29 993   | `datasets/euroc/MH_02_easy/gt_tum.txt` |
| `MH_03_medium` | 1.11     | 2 701      | 27 008      | 26 302   | `datasets/euroc/MH_03_medium/gt_tum.txt` |
| `V1_01_easy`   | 1.15     | 2 912      | 29 120      | 28 711   | `datasets/euroc/V1_01_easy/gt_tum.txt` |
| `V2_01_easy`   | 0.88     | 2 281      | 22 800      | 22 401   | `datasets/euroc/V2_01_easy/gt_tum.txt` |

GT → TUM conversion (one-liner, EuRoC ns + `[qw,qx,qy,qz]` → TUM s + `[qx,qy,qz,qw]`):

```
awk -F',' 'NR>1 { printf "%.9f %s %s %s %s %s %s %s\n", $1/1e9, $2, $3, $4, $6, $7, $8, $5 }' \
  <SEQ>/mav0/state_groundtruth_estimate0/data.csv > <SEQ>/gt_tum.txt
```

ATE evaluation uses Sim(3) alignment (mono scale is unobservable without explicit VIO metric output):

```
evo_ape tum <SEQ>/gt_tum.txt build/trajectory.txt --align --correct_scale --t_max_diff 0.05 --no_warnings
```

Run flags that worked:

- Mono VIO: `./build/run_mono --euroc datasets/euroc/<SEQ> --accel --reference-policy heuristic --skip-frames 0 --no-viz --repro-eval`
- Visual-only baseline: same command minus `--accel`.

stereo baseline auto-loaded from sensor.yaml: `0.110074 m` on MH_01.
Passing the inner `mav0` directory is wrong for this loader (`EurocDataset` already appends `/mav0/...`).

### 2.8 EuRoC Mono `--accel` VIO Sweep (2026-04-20)

Measured on the five sequences above, full length (no `--max-frames`), Sim(3) alignment via `evo_ape`.

| Sequence       | Frames | Visual only | +VIO      | Δ mean | Δ max  | VI init |
| ---            | ---:   | ---:        | ---:      | ---:   | ---:   | ---     |
| `MH_01_easy`   | 3 683  | 3.44 m      | **1.41 m** | **−59 %** | −43 % | rejected (rot_rms 0.13) |
| `MH_02_easy`   | 3 041  | 2.64 m      | **1.72 m** | **−35 %** | −42 % | rejected |
| `MH_03_medium` | 2 701  | **2.48 m**  | 2.91 m    | **+17 %** | +59 % | rejected |
| `V1_01_easy`   | 2 912  | 1.25 m      | **1.31 m** | +5 %   | **−59 %** | **succeeded** (scale 0.98) |
| `V2_01_easy`   | 2 281  | 0.69 m      | **0.63 m** | **−9 %**  | **−53 %** | rejected (scale outside tolerance) |
| **Average**    |        | **2.10 m**  | **1.60 m** | **−24 %** | —     | 1 / 5 accept |

Read these numbers as:

- **Machine Hall (long flight, translation-dominant):** IMU wins decisively on MH_01/02 because visual drift accumulates over distance and IMU anchors orientation. MH_03 is the open regression (§8.11).
- **Vicon Room (short, rotation-heavy):** the mean barely moves but the worst-case excursion (`max`) drops 50–60 %, i.e. IMU kills the occasional tracking runaway.
- **VI init acceptance is the second-order story, not the first.** The 9-DoF preintegration residual alone delivers the bulk of the gain. VI init just adds a small scale + gravity refinement on the one sequence (V1_01) whose early-mono rotations are clean enough to pass the 0.08 rad residual threshold.

Machine-readable artifact: none yet — the numbers above came from ad-hoc runs during the 0c.e validation. A proper `eval/euroc_vio_sweep.md` / `.json` is a pending artifact (§11).

Knob sweep finding (MH_01 only): `SVSLAM_BA_VELOCITY_PRIOR_SIGMA_M` ∈ {0.1, 0.3, 0.5, 1.0}, σ=0.3 is optimal (1.73 m vs 3.70 / 2.02 / 2.05 m). Tighter over-constrains; looser wastes IMU signal. σ=0.3 is the default.

### 2.9 600-Frame Loop Stability (historical)

There are two different pieces of evidence in the repo, and they should not be conflated:

1. Older explicit artifact in `eval/stella_comparison_results.md`:

| Run | ATE (m) |
| --- | ---: |
| `rep1` | `0.13731232` |
| `rep2` | `0.61716193` |
| `rep3` | `0.87081246` |
| median | `0.61716193` |

This artifact is from the older loop-enabled room-depth validation and shows that long-horizon stability was still weak.

2. Later retained project note from the post-`c5bcbe1` tuning:

- after raising `loop_cooldown_kf_` from `120` to `200`, later 600-frame reruns were recorded as roughly **`0.109 m`** and **`0.124 m`**
- those later numbers were kept in previous planning notes but were **not** written into a fresh standalone `eval/*.md` artifact on current HEAD

Current truth:

- the old `eval/stella_comparison_results.md` 600-frame median (`~0.618 m`) remains a **historical** warning, not the current behavior
- a **single-source** loop-enabled 600-frame room-depth measurement is now in **`eval/room_depth_600frame_report.md`** (mean ATE `~0.110 m` on `freiburg1_room`, 600 frames, async + loop closing, 2026-04-15)
- treat that file’s command line and SHA as the reproducible anchor; rerun after material pose-graph or loop-closing changes

---

## 3. Commit History

Two spans matter for this handoff.

**Span A — since last merged PR (`3f9bc71` → `master` HEAD `46a726d`):** 10 commits introducing VIO Stage 0b + 0c. Subsequent cleanup commits (`688f748`, `8e6f4e7`) live on the `vio-integration` branch / PR #3.

| Commit     | Stage  | What changed |
| ---        | ---    | --- |
| `38073b6`  | 0b.b   | `core/` velocity + IMU-bias scaffolding on `Frame` / `Keyframe` |
| `2f9bbf7`  | 0b.a   | `sensors/imu_preintegrator.h` (Forster-style skeleton) |
| `3b90a77`  | 0b.d   | `ImuEntry` type + EuRoC `imu0` loader |
| `03bd424`  | 0b.c   | Plumb EuRoC IMU into Tracking via `imu_buffer_` |
| `06868c5`  | pre-0c | Guard `Frame::landmarks_` and `Keyframe::landmarks_` container races |
| `bebc7d5`  | 0b.e / 0b.f / 0c.b | Tracking-side IMU path (`predictVelocityFromImu`, `reconcileVelocityWithVisual`), EuRoC `cam0 T_BS` extrinsic, mono/depth init gravity transform into camera frame |
| `f9303ac`  | 0c.c / 0c.a | `VelocityPreintegrationError` (Forster 6-DoF pos + vel), per-KF velocity + accel/gyro bias BA parameter blocks, `BiasAnchorError` + `BiasRandomWalkError` |
| `935a33c`  | 0c.d   | 3-DoF rotation residual added (9-DoF total), `delta_R` kept frozen (no gyro Jacobian yet) |
| `bd7b691`  | 0c.e   | `VisualInertialInitializer` (linear 2-stage solve), `tryVisualInertialInit` in Tracking, `applyGyroBiasCorrectionToSpans`, acceptance thresholds (rot_rms ≤ 0.08 rad, gyro-bias cap 0.05 rad/s) |
| `46a726d`  | docs   | `plan.md`: record Stage 0b/0c progress (this file was the previous update) |

On `vio-integration` branch (PR #3):

| Commit    | What changed |
| ---       | --- |
| `688f748` | `tests/test_euroc_dataset.cc`: unique temp paths across `ctest -j` workers (PID + atomic counter + nanosecond timestamp) |
| `8e6f4e7` | `backend/optimizer.cc`: env-tunable gravity prior weight (`SVSLAM_BA_GRAVITY_PRIOR_WEIGHT`, default 2.0) |

**Span B — earlier context preserved for archaeology:**

| Commit | What changed |
| --- | --- |
| `d3c81a7` | Improved tracking accuracy with stricter ratio tests, BA iteration tuning, and loop-correction ordering changes |
| `6429d9b` | Refactored core SLAM modules and expanded the test suite substantially |
| `244bb56` | Stabilized loop closing, added the first `stella_vslam` comparison, updated planning notes |
| `6d81697` | Added `MetricDepthEstimator`, improved tracking, completed the initial stella comparison |
| `c5bcbe1` | Improved monocular initialization, stabilized 600-frame loops, refreshed README |
| `c61d8b1` | Tightened regression baselines and documented the room-gap design problem |
| `f440bdc` | Increased local BA iterations from `10` to `15` |
| `8007ee2` | Added covisibility-weighted pose-graph edges and documented metric-depth testing |
| `a32d904` | Fixed metric-depth estimator crashes on dynamic-output-shape ONNX models |
| `d7b4657` | Improved `room_mono` tracking, verified covisibility + metric depth, updated plan |
| `c8e85a1` | Added EuRoC stereo scaffolding, CI smoke regression, and metric-depth verification |
| `e47469a` | Released `0.2.0`: stereo depth, ROS2 node, CI fix |
| `173a8d9` | Fixed ROS2 build, verified EuRoC stereo, improved pose-graph optimizer |
| `b5f2eea` | Verified stereo SLAM pipeline using synthetic EuRoC data |
| `121e044` | Revamped README with demo images, quick start, and architecture diagram |

---

## 4. Codebase Overview

### 4.1 Important Root Files

These are outside the directory inventory requested below but matter immediately:

- `CMakeLists.txt`: project version `0.2.0`, optional `USE_DBOW2`, optional `USE_DEPTH_DL`, test wiring
- `README.md`: public-facing summary; useful, but not always numerically current
- `CHANGELOG.md`: `0.2.0` release notes
- `CONTRIBUTING.md`: contributor/build expectations
- `RELEASING.md`: release checklist
- `CITATION.cff`: citation metadata

### 4.2 Full Inventory With Line Counts And Roles

#### `.github/workflows/`

- `ci.yml` (`61`): Ubuntu/Ninja CI; builds, runs `ctest`, `run_mono --help`, Python smoke checks

#### `apps/`

- `run_mono.cc` (`700`): main CLI runner; parses all flags, loads datasets, creates `Map` / `Tracking` / `LocalMapping` / `LoopClosing`, writes trajectories, saves `map.bin`, writes run-summary JSON

#### `config/examples/`

- `euroc_mh01.json` (`20`): sample EuRoC stereo pinhole override JSON
- `tum_pinhole_fr1.json` (`9`): sample TUM pinhole override JSON

#### `src/core/`

- `camera.cc` (`50`): pinhole project / unproject implementation
- `camera.h` (`28`): pinhole camera definition
- `common.h` (`33`): common typedefs, Eigen/Sophus aliases, shared includes
- `frame.cc` (`52`): frame pose setters/getters, ORB extraction, depth helpers
- `frame.h` (`56`): live frame container with image, pose, descriptors, landmarks, depth
- `heuristic_reference_keyframe_policy.cc` (`58`): current default reference-keyframe policy
- `heuristic_reference_keyframe_policy.h` (`14`): heuristic policy declaration
- `keyframe.cc` (`118`): keyframe copy-from-frame, covisibility update, neighbor ranking
- `keyframe.h` (`50`): keyframe state, covisibility graph, depth/gravity storage
- `landmark.cc` (`28`): landmark position and observation updates
- `landmark.h` (`39`): landmark definition, descriptor, observation map, mutex
- `map.cc` (`41`): add/remove/clear keyframes and landmarks
- `map.h` (`34`): global map container and `loop_correcting_` atomic
- `reference_keyframe_policy.h` (`45`): policy interface and decision types

#### `src/tracking/`

- `initializer.cc` (`557`): monocular two-view initialization, H/F model selection, triangulation
- `initializer.h` (`42`): initializer API and result containers
- `tracking.cc` (`~2720`): front-end tracking, motion model, keyframe decision, local-map tracking, relocalization, reinitialization, loop-correction handoff, **IMU velocity prediction (`predictVelocityFromImu`), visual-velocity reconciliation (`reconcileVelocityWithVisual`), preintegration span attachment (`populateKeyframeImuSpan`), VI init entry (`tryVisualInertialInit`)**
- `tracking.h` (`~200`): tracking state, recovery state, loop-correction state, run statistics, thresholds, **`T_cam_imu_` extrinsic, VI init bookkeeping**
- **`visual_inertial_initializer.cc` (`~400`):** linear two-stage VI init (closed-form gyro bias with cap, LSQ for scale + gravity + velocities, span delta_R first-order correction via `applyGyroBiasCorrectionToSpans`)
- **`visual_inertial_initializer.h` (`~165`):** `Options` / `Result` structs, acceptance thresholds

#### `src/backend/`

- `local_mapping.cc` (`449`): local-mapping queue, new-point creation, map-point culling, local BA
- `local_mapping.h` (`62`): local-mapping API, queue, callback hook
- `optimizer.cc` (`~1250`): pose-only PnP refinement, local BA, depth prior, gravity prior (env-tunable weight), pose graph, IRLS, **VIO preintegration residual wiring (velocity + accel/gyro bias blocks, bias anchor + random-walk priors, preint-residual gate for VI-init-aware datasets)**
- `optimizer.h` (`~390`): Ceres residual definitions and optimizer API, **`VelocityPreintegrationError` (Forster 9-DoF pos + vel + rot with first-order accel-bias Jacobian), `VelocityDeltaPriorError` (loose fallback), `BiasAnchorError`, `BiasRandomWalkError`**

#### `src/loop_closing/`

- `loop_closing.cc` (`911`): candidate search, Sim3 RANSAC/refinement, loop-edge weighting, stale-edge decay, pose-graph correction, landmark fusion
- `loop_closing.h` (`141`): loop-closing API, thresholds, loop-constraint data structures

#### `src/depth/`

- `depth_estimator.h` (`26`): depth estimator interface
- `metric_depth_estimator.cc` (`565`): ONNX Runtime metric-depth inference, dynamic-shape handling, input/output tensor discovery
- `metric_depth_estimator.h` (`77`): metric-depth estimator API
- `onnx_depth_estimator.cc` (`181`): relative-depth ONNX estimator
- `onnx_depth_estimator.h` (`38`): relative-depth estimator API
- `stereo_depth_estimator.cc` (`113`): StereoSGBM depth computation
- `stereo_depth_estimator.h` (`34`): stereo-depth estimator API and min/max depth constants

#### `src/sensors/`

- `accelerometer.h` (`80`): accelerometer entry type and simple processing helpers
- `imu.h` (`15`): `ImuEntry` (accel + gyro + timestamp) value type
- `imu_preintegrator.h` (`~110`): Forster-style IMU preintegration (`deltaR/deltaV/deltaP/dt`, bias reset, `integrate`, `predict` with gravity)
- `imu_preintegration_span.h` (`~40`): frozen per-KF-pair preintegration span (deltas, reference biases, `T_cam_imu` snapshot, `from_kf_id`, validity)

#### `src/io/`

- `euroc_dataset.cc` (`~495`): EuRoC dataset loader, stereo pairing, calibration setup, baseline extraction, **`imu0/data.csv` loader, multi-line `T_BS` YAML folding, `cam0FromImuExtrinsic` SE3 publishing with SVD re-orthonormalization**
- `euroc_dataset.h` (`~100`): EuRoC dataset API, **`allImu()` / `getImuBetween()` / `hasCam0FromImuExtrinsic()`**
- `euroc_pinhole_calibration.cc` (`188`): EuRoC JSON calibration parser
- `euroc_pinhole_calibration.h` (`34`): EuRoC calibration structs/API
- `map_io.cc` (`269`): map save/load
- `map_io.h` (`18`): map I/O API
- `tum_dataset.cc` (`263`): TUM RGB-D/accelerometer loader and timestamp association
- `tum_dataset.h` (`66`): TUM dataset API
- `tum_pinhole_calibration.cc` (`124`): TUM JSON calibration parser
- `tum_pinhole_calibration.h` (`27`): TUM calibration structs/API

#### `src/experiments/reference_keyframe/` (runtime-relevant even though not in the original requested list)

- `pipeline_reference_keyframe_policy.cc` (`111`): staged-gates experimental policy
- `pipeline_reference_keyframe_policy.h` (`21`): staged-gates policy declaration
- `score_reference_keyframe_policy.cc` (`89`): weighted-score experimental policy
- `score_reference_keyframe_policy.h` (`20`): weighted-score policy declaration

#### `tests/`

- 18 files total, `ctest -j4` reports 77 / 77 PASS
- `test_camera.cc` (`59`): camera projection/unprojection tests
- `test_euroc_dataset.cc` (`~260`): EuRoC dataset and stereo-calibration tests, **`cam0 T_BS` single-line and multi-line YAML, IMU CSV**, PID + nanosecond-unique temp paths
- `test_imu_preintegrator.cc` (`~90`): zero-motion identity, constant-accel `deltaV`/`deltaP`, `predict` with gravity, bias subtraction, reset
- `test_frame.cc` (`90`): frame depth and backprojection tests
- `test_initializer.cc` (`29`): initializer smoke/regression tests
- `test_keyframe.cc` (`74`): covisibility ranking and connection tests
- `test_landmark.cc` (`80`): landmark CRUD and thread-safety tests
- `test_loop_closing.cc` (`83`): loop weighting and stale-edge decay tests
- `test_map.cc` (`96`): map add/remove/concurrency tests
- `test_metric_depth_estimator.cc` (`40`): metric-depth tensor-shape/model tests
- `test_optimizer.cc` (`~345`): BA, pose graph, depth prior, gravity prior tests, **VelocityDelta prior, 9-DoF preintegration residual (match, gravity, accel-bias first-order, rotation match, rotation mismatch), bias anchor, bias random-walk**
- `test_visual_inertial_initializer.cc` (`~255`): scale + gravity recovery on synthetic EuRoC-like scene, capped closed-form gyro-bias recovery, metric-scale mode, missing-span rejection
- `test_reference_keyframe_policy.cc` (`88`): heuristic/score/pipeline policy tests
- `test_stereo_depth_estimator.cc` (`87`): stereo-depth correctness tests
- `test_synthetic_scene.h` (`79`): shared synthetic-scene helper for tests
- `test_tracking.cc` (`84`): tracking state-machine tests
- `test_tracking_pose_recompute.cc` (`57`): recompute-pose acceptance and recovery-window tests
- `test_tracking_run_statistics.cc` (`14`): run-statistics defaults test
- `test_tum_pinhole_calibration.cc` (`52`): TUM calibration loader tests

#### `scripts/`

- `build_leaderboard.py` (`218`): method × sequence benchmark matrix and ranking
- `check_regression_gate.py` (`128`): deterministic trajectory + ATE ceiling regression gate runner
- `download_ci_test_data.sh` (`8`): CI synthetic-data download helper
- `eval_all.sh` (`341`): broader batch-evaluation harness
- `eval_ate.sh` (`53`): thin `evo_ape` wrapper
- `eval_lib.py` (`100`): shared Python evaluation helpers
- `eval_reference_policies.sh` (`200`): reference-policy evaluation harness
- `extract_keyframe_trajectory.py` (`68`): extract keyframe trajectories
- `generate_ci_test_data.py` (`232`): generate deterministic synthetic CI data
- `generate_map_report.py` (`756`): map-report generation
- `generate_tum_report.py` (`734`): TUM run-report generation
- `print_ate_mean.py` (`40`): print mean ATE from GT + trajectory
- `run_ci_smoke_regression.py` (`143`): CI smoke regression runner
- `update_reference_policy_docs.py` (`959`): sync/reference-policy docs generation
- `verify_comparison_benchmark.sh` (`85`): run standard comparison presets
- `__pycache__/build_leaderboard.cpython-312.pyc` (`BIN`): generated Python bytecode cache
- `__pycache__/check_regression_gate.cpython-312.pyc` (`BIN`): generated Python bytecode cache
- `__pycache__/eval_lib.cpython-312.pyc` (`BIN`): generated Python bytecode cache
- `__pycache__/generate_ci_test_data.cpython-312.pyc` (`BIN`): generated Python bytecode cache
- `__pycache__/generate_map_report.cpython-312.pyc` (`BIN`): generated Python bytecode cache
- `__pycache__/print_ate_mean.cpython-312.pyc` (`BIN`): generated Python bytecode cache
- `__pycache__/run_ci_smoke_regression.cpython-312.pyc` (`BIN`): generated Python bytecode cache

#### `eval/`

- `comparison_protocol.md` (`93`): rules for fair external OSS comparison
- `demo_comparison_xyz.png` (`BIN`): demo comparison image
- `demo_map_xyz.html` (`534`): generated demo map report
- `demo_trajectory_room.png` (`BIN`): demo room trajectory image
- `demo_trajectory_xyz.png` (`BIN`): demo xyz trajectory image
- `euroc_mono_summary.json` (`1`): latest EuRoC mono run summary
- `euroc_stereo_run_results.md` (`34`): stereo run notes
- `euroc_stereo_summary.json` (`1`): latest EuRoC stereo run summary
- `euroc_test_results.md` (`135`): EuRoC verification report
- `leaderboard_suite.json` (`70`): benchmark suite definition
- `metric_depth_test_results.md` (`217`): metric-depth pipeline verification
- `regression_baselines.json` (`61`): five regression-gate ceilings and protocol args
- `stella_comparison.json` (`126`): machine-readable comparison data
- `stella_comparison_results.md` (`96`): human-readable `stella_vslam` comparison report

#### `docs/`

- `decisions.md` (`38`): public decision log around reference-keyframe policy work
- `experiments.md` (`120`): public experiment tables
- `images/demo_comparison_xyz.png` (`BIN`): public demo asset
- `images/demo_trajectory_room.png` (`BIN`): public demo asset
- `images/demo_trajectory_xyz.png` (`BIN`): public demo asset
- `index.md` (`32`): public landing page
- `interfaces.md` (`36`): public interface notes

#### `ros2/`

- `BUILD_LOG.md` (`150`): recorded ROS2 build log
- `CMakeLists.txt` (`51`): ROS2 package build wiring
- `README.md` (`59`): ROS2 usage guide
- `launch/slam.launch.py` (`57`): ROS2 launch file
- `launch/__pycache__/slam.launch.cpython-312.pyc` (`BIN`): generated Python bytecode cache
- `package.xml` (`26`): ROS2 package manifest
- `src/slam_node.cc` (`381`): basic ROS2 node wrapping `Tracking` + `LocalMapping` only

---

## 5. Thread Model

```mermaid
flowchart LR
    A[run_mono main thread] --> B[Tracking::addFrame / track]
    B -->|insertKeyframe| C[LocalMapping queue]
    C --> D[LocalMapping thread]
    D -->|insertKeyframe| E[LoopClosing queue]
    E --> F[LoopClosing thread]
    D -->|on_ba_completed_| B
    F -->|on_loop_corrected_| B
    B --> G[(Map / Keyframes / Landmarks)]
    D --> G
    F --> G
```

Important execution modes:

- **Default async mode:** main thread runs tracking; local mapping and loop closing are background threads
- **`--repro-eval` mode:** local mapping runs synchronously via `processPendingWork()` and loop closing is disabled
- **ROS2 node:** currently runs `Tracking` + `LocalMapping`; it does **not** start `LoopClosing`

Code anchors:

- thread startup/shutdown: `apps/run_mono.cc:321-400`, `apps/run_mono.cc:628-638`
- local-mapping loop: `src/backend/local_mapping.cc:29-57`
- loop-closing loop: `src/loop_closing/loop_closing.cc:403-421`

---

## 6. Thread Safety

| Shared data | Writers | Readers | Guard / contract | Notes |
| --- | --- | --- | --- | --- |
| `Map::keyframes_`, `Map::landmarks_` | tracking, local mapping, loop closing, map I/O | all modules | `Map::mutex_` for add/remove only | `getAllKeyframes()` / `getAllLandmarks()` return const refs without locking; snapshot immediately |
| `Map::loop_correcting_` | loop closing | tracking, local mapping | `std::atomic<bool>` | used as a global pause flag during pose-graph correction |
| `Frame::T_cw_` | tracking and callbacks | tracking | `Frame::mutex_` via `setPose()` / `getPose()` | current frame callback interactions are further serialized by `Tracking::pose_mutex_` |
| `Tracking::loop_correction_state_`, `current_frame_` callback path | tracking thread, loop/BA callbacks | tracking thread | `Tracking::pose_mutex_` | prevents worker threads from mutating the live frame directly |
| `Keyframe::T_cw_`, `landmarks_`, `connected_keyframes_` | local mapping, loop closing, optimizer | tracking, loop closing, optimizer | `Keyframe::mutex_` | loop-closing and optimizer snapshot these under lock |
| `Landmark::pos_w_`, `observations_` | local mapping, loop closing, optimizer | tracking, optimizer | `Landmark::mutex_` | tests cover basic thread safety |
| `LocalMapping::new_keyframes_` | tracking | local-mapping thread | `mutex_new_keyframes_` + `cv_new_keyframes_` | producer/consumer queue |
| `LoopClosing::new_keyframes_` | local mapping | loop-closing thread | `mutex_new_keyframes_` + `cv_new_keyframes_` | producer/consumer queue |

Current safety caveat:

- `Map::getAllKeyframes()` and `Map::getAllLandmarks()` are API-level weak spots because they expose internal containers by reference without holding `Map::mutex_`

---

## 7. Constants And Thresholds

These are the hard-coded runtime knobs that matter most. They are the first places to inspect before proposing architecture work.

### 7.1 Tracking / Recovery

| Location | Constant / threshold | Value | Meaning |
| --- | --- | ---: | --- |
| `src/tracking/tracking.h:129` | `max_lost_frames_` | `30` frames | give-up threshold for prolonged loss |
| `src/tracking/tracking.h:131` | `max_loop_correction_deferrals_` | `6` | max times to defer a pending loop correction |
| `src/tracking/tracking.h:132` | `min_loop_correction_correspondences_` | `80` | minimum live frame-landmark pairs before applying pending loop correction |
| `src/tracking/tracking.h:133` | `recovery_stabilization_window_frames_` | `3` | frames of stricter recovery gating after relocalization/loop handoff |
| `src/tracking/tracking.h:134` | `loop_relocalization_radius_m_` | `2.5 m` | tighter relocalization radius while loop correction is pending |
| `src/tracking/tracking.h:135` | `recovery_relocalization_radius_m_` | `4.0 m` | relocalization radius during general stabilization |
| `src/tracking/tracking.h:136` | `min_stable_support_` | `120` | support floor for accepting local-map pose updates during stabilization |
| `src/tracking/tracking.h:137` | `recovery_max_change_strict_` | `0.12` | strict translation/rotation change gate |
| `src/tracking/tracking.h:138` | `recovery_max_change_relaxed_` | `0.18` | relaxed translation/rotation change gate |
| `src/tracking/tracking.h:144` | `reinit_trigger_frames_` | `20` frames | start reinitialization after this many lost frames |
| `src/tracking/tracking.cc:18` | `kMaxDepthLandmarkMeters` | `10.0 m` | max depth used when creating depth-backed landmarks |
| `src/tracking/tracking.cc:19` | `kMinTrackedDepthMeters` | `0.15 m` | min positive depth for projected landmarks |
| `src/tracking/tracking.cc:20` | `kMaxTrackedDepthMeters` | `18.0 m` | max tracked depth for projection gating |
| `src/tracking/tracking.cc:21` | `kMaxIndoorCameraPositionMeters` | `50.0 m` | absolute camera-position sanity bound |
| `src/tracking/tracking.cc:22` | `kMinTrackLocalMapLandmarks` | `250` | minimum local-map landmark pool before global fallback |
| `src/tracking/tracking.cc:23` | `kMinBootstrapCorrespondences` | `30` | fallback bootstrap threshold |
| `src/tracking/tracking.cc:24` | `kMaxBootstrapMatches` | `200` | cap for fallback global descriptor matches |
| `src/tracking/tracking.cc:25` | `kMinTrackLocalMapInliers` | `12` | min inliers for local-map PnP acceptance |
| `src/tracking/tracking.cc:26` | `kMinTrackReferenceInliers` | `15` | min inliers for reference-frame PnP acceptance |
| `src/tracking/tracking.cc:27` | `kMinPoseRecomputeCorrespondences` | `10` | min correspondences for post-BA pose recompute |
| `src/tracking/tracking.cc:28` | `kMinPoseRecomputeInliers` | `20` | min inliers for post-BA pose recompute |
| `src/tracking/tracking.cc:29` | `kMaxDepthLandmarksPerKeyframe` | `600` | cap on new depth landmarks when inserting a keyframe |
| `src/tracking/tracking.cc:791` | `min_frames_since_last_kf` | `3` | earliest new-keyframe insertion |
| `src/tracking/tracking.cc:797` | `min_tracked_threshold` | `60` | low-tracking trigger for keyframe insertion |
| `src/tracking/tracking.cc:804` | `max_frames_since_last_kf` | `12` | forced keyframe insertion age |
| `src/tracking/tracking.cc:818` | tracked/reference ratio gate | `< 0.65` | another keyframe insertion trigger |

### 7.2 Tracking Match / PnP Gates

| Location | Threshold | Value | Meaning |
| --- | --- | ---: | --- |
| `src/tracking/tracking.cc:1340-1341` | frame-to-frame descriptor gate | `65` and `0.70` Lowe ratio | used by `trackReferenceKeyframe()` |
| `src/tracking/tracking.cc:1397` | reference projection gate | `48 px * (1 + 0.12*octave)` | projected-landmark acceptance in `trackReferenceKeyframe()` |
| `src/tracking/tracking.cc:1433` | reference PnP iterations / reproj / confidence | `250`, `8.0 px`, `0.995` | `solvePnPRansac()` for frame-to-frame PnP |
| `src/tracking/tracking.cc:1442` | reference refine inlier gate | `6.0 px` | `refinePnPInliers()` threshold |
| `src/tracking/tracking.cc:1464-1465` | reference pose-change gate | `0.35 m`, `0.45 rad` | reject large pose jumps |
| `src/tracking/tracking.cc:994` | local-map search radius | `100 px` | search area around projected landmark |
| `src/tracking/tracking.cc:1014` | local-map ratio gate | `0.7` Lowe ratio | projection search descriptor filter |
| `src/tracking/tracking.cc:1033` | initial local-map pose gate | `35 px` | filter correspondences by current pose |
| `src/tracking/tracking.cc:1047` | all-landmark fallback trigger | `< 80` visible landmarks | switch to wider bootstrap pool |
| `src/tracking/tracking.cc:1082-1084` | fallback descriptor gate | `65` and `0.75` Lowe ratio | global descriptor bootstrap |
| `src/tracking/tracking.cc:1115` | fallback pose gate | `55 px` or `180 px` | fallback pose-based pruning; larger when using all landmarks |
| `src/tracking/tracking.cc:1227` | local-map PnP iterations / reproj / confidence | `150`, `10.0 px`, `0.995` | `solvePnPRansac()` in `trackLocalMap()` |
| `src/tracking/tracking.cc:1246` | local-map refine inlier gate | `8.0 px` | `refinePnPInliers()` threshold |
| `src/tracking/tracking.cc:1266-1267` | local-map pose-change hard gate | `0.5 m`, `0.6 rad` | reject large post-PnP jumps |

### 7.3 Initializer

| Location | Constant | Value | Meaning |
| --- | --- | ---: | --- |
| `src/tracking/initializer.cc:12` | `kInitLoweRatio` | `0.75` | descriptor filter during initialization |
| `src/tracking/initializer.cc:13` | `kInitMedianParallaxDeg` | `0.8 deg` | minimum median parallax for F-based reconstruction |
| `src/tracking/initializer.cc:14` | `kInitHomographyMinParallaxDeg` | `1.0 deg` | minimum median parallax for H-based reconstruction |
| `src/tracking/initializer.cc:15` | `kInitPointParallaxDeg` | `0.35 deg` | minimum per-point bearing parallax |
| `src/tracking/initializer.cc:16` | `kInitMaxReprojErrorPx` | `3.0 px` | per-view reprojection ceiling in H reconstruction |
| `src/tracking/initializer.cc:17` | `kInitMinTriangulatedPoints` | `70` | minimum triangulated points to accept initialization |
| `src/tracking/initializer.cc:93` | min initialization matches | `100` | fail early below this |
| `src/tracking/initializer.cc:109-111` | H/F selector | `score_H / (score_H + score_F) > 0.50` | use homography above this ratio |

### 7.4 Local Mapping

| Location | Constant / threshold | Value | Meaning |
| --- | --- | ---: | --- |
| `src/backend/local_mapping.cc:174` | direct depth landmark max depth | `10.0 m` | depth-backed point creation |
| `src/backend/local_mapping.cc:196` | neighbor count for triangulation | `15` | covisibility neighbors when creating new points |
| `src/backend/local_mapping.cc:247` | mono min baseline | `0.02 m` | reject very short-baseline mono pairs |
| `src/backend/local_mapping.cc:247` | depth/stereo min baseline | `0.01 m` | reject very short-baseline depth pairs |
| `src/backend/local_mapping.cc:269-270` | triangulation descriptor gate | `64` and `0.8` Lowe ratio | new-point matching |
| `src/backend/local_mapping.cc:334-335` | triangulation depth range | `0.1 m` to `20.0 m` | reject invalid point depths |
| `src/backend/local_mapping.cc:348` | triangulation reprojection gate | `8.0 px` | reject noisy new map points |
| `src/backend/local_mapping.cc:394` | local BA covisibility window | `15` neighbors | keyframes included in local BA |
| `src/backend/local_mapping.cc:422` | `max_ba_landmarks` | `800` | local BA landmark cap |
| `src/backend/local_mapping.cc:441` | local BA iterations | `15` | current BA iteration count |

### 7.5 Covisibility / Reference-Policy Thresholds

| Location | Constant | Value | Meaning |
| --- | --- | ---: | --- |
| `src/core/keyframe.cc:67` | covisibility connection floor | `> 15` shared landmarks | only strong connections survive |
| `src/core/heuristic_reference_keyframe_policy.cc:10` | `kMinTrackedFeatures` | `35` | heuristic sparse-mono veto |
| `src/core/heuristic_reference_keyframe_policy.cc:11` | `kMinDetectedKeypoints` | `150` | heuristic sparse-mono veto |
| `src/experiments/reference_keyframe/score_reference_keyframe_policy.cc:17` | `kPromoteThreshold` | `0.55` | score-policy promotion cutoff |
| `src/experiments/reference_keyframe/pipeline_reference_keyframe_policy.cc:9-16` | mono/depth/staleness floors | `32`, `140`, `35`, `28`, `20`, `10`, `50`, `220` | staged-gates policy thresholds |

### 7.6 Loop Closing

| Location | Constant / threshold | Value | Meaning |
| --- | --- | ---: | --- |
| `src/loop_closing/loop_closing.h:123` | `min_loop_interval_kf_` | `30` | ignore near-temporal candidates |
| `src/loop_closing/loop_closing.h:124` | `max_loop_candidates_` | `4` | BoW shortlist size |
| `src/loop_closing/loop_closing.h:125` | `min_loop_score_` | `0.01` | BoW score floor |
| `src/loop_closing/loop_closing.h:126` | `min_loop_inliers_` | `30` | generic loop-support floor |
| `src/loop_closing/loop_closing.h:127` | `correction_window_size_` | `30` | correction-window size parameter |
| `src/loop_closing/loop_closing.h:128` | `loop_cooldown_kf_` | `200` | cooldown after a successful loop |
| `src/loop_closing/loop_closing.h:129` | `sim3_ransac_iterations_` | `200` | Sim3 RANSAC iterations |
| `src/loop_closing/loop_closing.h:130` | `max_sim3_residual_` | `0.25` | geometric inlier threshold |
| `src/loop_closing/loop_closing.h:131-132` | generic Sim3 scale range | `0.7` to `1.4` | scale search range without metric depth |
| `src/loop_closing/loop_closing.h:133` | overlap window | `140 KF` | overlapping loop-edge decay window |
| `src/loop_closing/loop_closing.cc:18` | default ratio matcher gate | `0.7` | generic helper |
| `src/loop_closing/loop_closing.cc:472` | `min_matches_for_loop` | `50` | descriptor matches needed before full verification |
| `src/loop_closing/loop_closing.cc:507,557,587` | candidate ratio gate | `0.8` | loop-candidate matching |
| `src/loop_closing/loop_closing.cc:643-644` | metric-depth scale bounds | `0.85` to `1.15` | tighter scale range when metric depth exists |
| `src/loop_closing/loop_closing.cc:646` | metric-depth relaxed residual | `0.35` | diagnostic relaxed residual |
| `src/loop_closing/loop_closing.cc:647` | metric-depth refine seed min | `15` | min RANSAC inliers before refinement |
| `src/loop_closing/loop_closing.cc:648` | metric-depth final min inliers | `22` | final geometric acceptance floor |
| `src/loop_closing/loop_closing.cc:649` | metric-depth final min inlier ratio | `0.38` | final ratio guard |
| `src/loop_closing/loop_closing.cc:650` | metric-depth scale tolerance | `0.05` | allowed final scale drift from `1.0` |
| `src/loop_closing/loop_closing.cc:255-263` | loop-edge weighting normalization | `inliers/40`, `ratio/0.60`, `3 + 4*confidence`, `scale=1000` in metric mode | loop-edge confidence policy |
| `src/loop_closing/loop_closing.cc:268-274` | stale-edge decay breakpoints | `0.15 m`, `0.02 scale`, floor `0.35` | stale loop-edge reuse decay |
| `src/loop_closing/loop_closing.cc:291-299` | overlap decay | `0.35`, `0.60`, `1.0` | overlapping loop-edge downweighting |

### 7.7 Optimizer / Depth

| Location | Constant / threshold | Value | Meaning |
| --- | --- | ---: | --- |
| `src/backend/optimizer.cc:188` | depth-prior max observed depth | `10.0 m` | ignore deeper depth observations |
| `src/backend/optimizer.cc:192` | depth sigma | `0.02` metric, `0.2` relative | stronger prior for metric depth |
| `src/backend/optimizer.cc:199` | depth prior loss | `Huber(0.5)` | robust depth residual |
| `src/backend/optimizer.cc:439` | landmark position bound | `100.0` | BA landmark bound |
| `src/backend/optimizer.cc:496` | gravity prior weight | `2.0` | soft gravity regularization |
| `src/backend/optimizer.cc:501` | gravity prior loss | `Huber(0.3)` | robust gravity residual |
| `src/backend/optimizer.cc:522` | local BA solver | `DENSE_SCHUR` | current BA linear solver |
| `src/backend/optimizer.cc:711` | covisibility scale weight in pose graph | `100.0` if `fix_scale`, else `1.5 * weight_scale` | scale regularization |
| `src/backend/optimizer.cc:753` | metric scale prior weight | `100.0` | log-scale prior when fixing scale |
| `src/backend/optimizer.cc:771-772` | pose-graph losses | loop `Cauchy(1.0)`, covisibility `Huber(1.0)` | robustification choice |
| `src/backend/optimizer.cc:782-785` | pose-graph solver and iterations | `SPARSE_NORMAL_CHOLESKY`, `SUITE_SPARSE`, up to `90` | current pose-graph backend |
| `src/backend/optimizer.cc:824` | IRLS cutoff | `median + 2.5 * MAD`, min `1.0` | loop-edge residual cutoff |
| `src/backend/optimizer.cc:834` | IRLS min weight | `0.20` | downweight floor |
| `src/depth/stereo_depth_estimator.h:20-21` | stereo depth range | `0.1 m` to `20.0 m` | valid stereo depth window |
| `src/depth/stereo_depth_estimator.cc:98-109` | StereoSGBM params | block `5`, uniqueness `10`, speckle `50`, range `2`, `disp12=1` | current stereo matcher setup |
| `src/depth/metric_depth_estimator.cc:84` | ONNX intra-op threads | `4` | metric-depth inference thread count |

### 7.8 VIO Env Knobs (all optional — defaults preserve the Stage 0c empirical state)

| Env variable | Default | Meaning |
| --- | ---: | --- |
| `SVSLAM_VIO_VELOCITY_IMU_ALPHA` | `0.3` | Blend weight for the IMU prediction vs the post-track visual pose delta inside `reconcileVelocityWithVisual`. 0 = pure visual, 1 = pure IMU open-loop |
| `SVSLAM_VIO_ENABLE_VISUAL_VELOCITY` | unset | Opt-in: populate `Frame::velocity_` on non-IMU runs (TUM) so BA's velocity prior acts as a motion-smoothness regularizer |
| `SVSLAM_BA_VELOCITY_PRIOR_SIGMA_M` | `0.3` (m) | Position-side sigma for the preintegration + loose-delta residuals. MH_01 sweep says this is optimal; `<=0` disables the velocity-related residuals entirely |
| `SVSLAM_BA_VELOCITY_PRIOR_VEL_SIGMA` | `0.3` (m/s) | Velocity-side sigma for the preintegration residual |
| `SVSLAM_BA_PREINT_ROT_SIGMA_RAD` | `0.05` (≈ 2.9°) | Rotation residual sigma per KF-gap. `<=0` zeros the rotation weight, restoring 6-DoF behavior |
| `SVSLAM_BA_BIAS_ACCEL_ANCHOR_SIGMA` | `0.5` (m/s²) | Accel-bias anchor sigma |
| `SVSLAM_BA_BIAS_GYRO_ANCHOR_SIGMA` | `0.1` (rad/s) | Gyro-bias anchor sigma |
| `SVSLAM_BA_BIAS_ACCEL_RW_SIGMA` | `0.05` (m/s² per pair) | Accel-bias random-walk sigma between consecutive KFs |
| `SVSLAM_BA_BIAS_GYRO_RW_SIGMA` | `0.005` (rad/s per pair) | Gyro-bias random-walk sigma |
| `SVSLAM_BA_GRAVITY_PRIOR_WEIGHT` | `2.0` | Per-KF gravity prior weight. `<=0` disables; useful on aggressive-motion sequences where the ±50 ms accel window is dominated by motion rather than gravity |
| `SVSLAM_VIO_MIN_INIT_KEYFRAMES` | `15` | Minimum KFs before `tryVisualInertialInit` attempts the linear solve |
| `SVSLAM_VIO_GATE_PREINT` | unset | Opt-in: restore the original "gate preintegration until VI init succeeds" behavior. Default is preint always on — safer on sequences where VI init never converges (e.g. MH_01) |
| `SVSLAM_VIO_FORCE_PREINT_BA` | unset | Legacy inverse of `SVSLAM_VIO_GATE_PREINT`, kept for backwards-compatibility with pre-2026-04-20 benchmark reruns |

Tuning philosophy: the defaults above were arrived at on EuRoC MH_01 with a four-point σ sweep; they leave the residuals noticeably *loose* so visual BA wins ties. Do not tighten `SVSLAM_BA_VELOCITY_PRIOR_SIGMA_M` below `0.2` without fresh evidence — MH_01 regressed 2× at σ=0.1.

---

## 8. Known Issues

Ranked by severity for **2026-04-21**:

1. **`room_mono` remains far behind `stella_vslam` and still lacks early loop detections.**
   - Current gate: `0.176506 m` (improved from `0.197374 m` after the pre-VIO work, 2026-04-16)
   - `stella_vslam`: `0.02743546 m`
   - Published loop-enabled comparison recorded `0/0/0` loop detections inside the first 250 frames.
   - **plan.md §10.4's ordered experiment #1 (ORB 2000 → 3000) was re-tested on 2026-04-20 and regresses both mono room gates — see §8.12.**

2. **`xyz_mono_*` gates pass with thin margins.**
   - `xyz_mono_head_repro`: `0.028136 m` / ceiling `0.030000 m` (7 % slack)
   - `xyz_mono_accel_head_repro`: `0.024931 m` / ceiling `0.027000 m` (7 % slack)
   - `room_mono_accel_head_repro`: `0.164001 m` / ceiling `0.170000 m` (4 % slack — thinnest)
   - Any further mono front-end change must run these three gates before and after.

3. **`room_depth` is still materially behind `stella_vslam`.**
   - Current strict repro gate: `0.079914 m`
   - `stella_vslam`: `0.02110508 m` (≈ 3.8× gap)
   - ORB 3000 experiment incidentally improved this to `0.068392 m` but was reverted because it broke room_mono — suggests there is room to move if mono damage can be avoided.

4. **600-frame loop documentation:** see **`eval/room_depth_600frame_report.md`** (2026-04-15). The older `~0.618 m` median in `eval/stella_comparison_results.md` remains historical context only.

5. **`Map::getAllKeyframes()` and `Map::getAllLandmarks()` expose internal containers without holding the map mutex.**
   - Code mostly snapshots immediately, but the API is still easy to misuse
   - See `src/core/map.cc:27-33`

6. **Metric depth works on `xyz` but is poor on `room` at 250 frames.**
   - `room 250`: `0.38429766 m`
   - Sensor depth on the same scenario is much better

7. **The ROS2 node is functional but not feature-parity with the CLI.**
   - No loop-closing thread is started in `ros2/src/slam_node.cc`
   - **No VIO hookup either** — `Tracking::imu_buffer_` / `Tracking::setImuToCameraExtrinsic` are unused in the ROS2 path

8. **The README "6k lines" claim is now stale.**
   - Current measured app+core count is `12 090`
   - Documentation issue, not an algorithmic blocker

9. **VI init rejects noisy-mono sequences without falling through to ORB-SLAM3-style MAP refinement.**
   - MH_01 and MH_03 both have rot_rms ≈ 0.13 rad during the first 15 KFs, above the 0.08 rad threshold
   - V2_01 rejects with "scale outside tolerance" (scale ≈ 0.007 — degenerate visual window)
   - Only V1_01 accepts; gain is modest (scale correction ≈ 2 %)
   - Follow-up: §16.5

10. **Gyro-bias first-order Jacobian inside the BA rotation residual was tried and reverted (2026-04-20).**
    - Adds a free parameter that absorbs visual rotation noise when biases are not pre-calibrated
    - Regressed MH_01 from `1.41 m` to `2.68–3.56 m` at any gyro-anchor σ in `[0.01, 0.1]`
    - **Do not re-enable without a VI-init stage that delivers gyro bias to O(0.01 rad/s) first**

11. **MH_03_medium is the open VIO regression.**
    - Visual-only `2.48 m` → +VIO `2.91 m` (+17 %)
    - Disabling preintegration residual (`SVSLAM_BA_VELOCITY_PRIOR_SIGMA_M=0`): `3.04 m` — still worse than visual-only
    - Disabling gravity prior (`SVSLAM_BA_GRAVITY_PRIOR_WEIGHT=0`): `2.90 m` — essentially no change
    - Hypothesis: MH_03's aggressive motion makes every IMU-path contribution noisy. The 126 `Lost` events vs 82 in the visual-only run suggest the first `predictVelocityFromImu` result or the per-KF gravity alignment from a ±50 ms aggressive-motion window is destabilizing Tracking. Not yet root-caused; see §16.4 for the next bisection to try.

12. **`plan.md` §10 ordered experiments are partially stale.**
    - §10.4 #1 ORB 2000 → 3000 regresses both mono room gates (`0.177 m` → `0.227 m`; `0.164 m` → `0.236 m`, above the `0.170 m` ceiling) despite plan.md predicting "strongest immediate increase in mono match density"
    - §10.4 #4 / §10.2 #1 (initializer H/F ratio `0.50 → 0.60`) is already in the code
    - Remaining items should be probed one-by-one on the narrowest relevant gate. See §12 for the recommended workflow

13. **EuRoC stereo-only depth is degraded (pre-existing, not a VIO regression).**
    - MH_01 `--stereo` no-IMU ATE ≈ `2.77 m` vs `3.44 m` visual mono (worse than mono on the long-distance flight)
    - Pinhole stereo rectification is not handling EuRoC's fisheye-ish distortion well
    - Unblocks the stereo tracking mode (Feature Matrix "Partial" row). Orthogonal to VIO work.

---

## 9. Roadmap

### Phase A: Numerical Quality Foundation

| Status | What is done | What remains |
| --- | --- | --- |
| Mostly done | deterministic replay, expanded tests, stricter gates, BA improvements, covisibility-weighted pose graph, recovered `xyz_mono` gate | keep `xyz_mono` under ceiling while improving `room_mono`; refresh current-HEAD external comparison numbers |

### Phase B: Learned Depth Differentiation

| Status | What is done | What remains |
| --- | --- | --- |
| Partial | relative depth, metric depth, dynamic-shape fix, indoor metric ONNX verification | make metric depth competitive on room-scale sequences; consider confidence-aware or temporal filtering |

### Phase C: Robustness / Thread Safety

| Status | What is done | What remains |
| --- | --- | --- |
| Partial | loop-correction safe-point handoff, stale-edge decay, overlap decay, mutex cleanup in several hot paths | map read/write discipline cleanup, better relocalization behavior |

### Phase D: Feature Expansion

| Status | What is done | What remains |
| --- | --- | --- |
| Partial | EuRoC loader, EuRoC stereo depth, ROS2 node, metric-depth CLI, VIO Stage 0b (IMU preintegration scaffolding), VIO Stage 0c (BA preintegration residual + bias BA params + VI init), empirical MH_01/V1_01 ATE validation | VIO Stage 0c.f (ORB-SLAM3-style MAP refinement, larger window, gyro-bias Jacobian once biases converge), stronger ROS2 parity, true stereo/tracking evolution |

#### VIO status (2026-04-20)

Commits `bebc7d5..bd7b691` landed the loosely-coupled VI pipeline. Validated on EuRoC mono + `--accel`:

| Dataset    | Visual only | +VIO (Stage 0c.d+e) |
| --- | ---: | ---: |
| MH_01_easy | 3.44 m      | **1.41 m** (−59%)   |
| V1_01_easy | 1.25 m      | **1.31 m**          |

- VI Init accepts V1_01 (rot_rms 0.05 rad < 0.08 threshold) and rejects MH_01 (rot_rms 0.13 rad). MH_01 benefits from the 9-DoF BA residual alone; V1_01 picks up additional gain from scale + gravity refinement.
- Loose sigma defaults: `SVSLAM_BA_VELOCITY_PRIOR_SIGMA_M=0.3` m, `SVSLAM_BA_PREINT_ROT_SIGMA_RAD=0.05` rad. MH_01 sweep shows σ=0.3 is optimal; tighter pulls poses too hard, looser discards IMU info.
- Tried but reverted: gyro-bias first-order Jacobian inside the BA rotation residual. It gives BA a free parameter that absorbs visual rotation noise when biases are uncalibrated, regressing MH_01 by 2–3×. Re-visit only after a reliable VI init calibrates gyro bias.

Known limitations:
- Mono + `--stereo` on EuRoC yields ATE ~2.8 m even visual-only — the pinhole-rectified stereo pipeline is not handling EuRoC's fisheye-ish distortion well. Separate from the VIO work.
- Gyro bias estimate from the linear VI init is unreliable on short EuRoC windows (O(0.05) rad/s vs O(0.004) ground truth); currently hard-capped at 0.05 rad/s and BA takes over via random-walk + anchor priors.

### Phase E: Community / Release Hygiene

| Status | What is done | What remains |
| --- | --- | --- |
| Mostly done | CI, CHANGELOG, semver, LICENSE, CONTRIBUTING, CITATION, public docs | refresh public numbers after the next benchmark pass; optionally add tutorial-style docs |

---

## 10. `stella_vslam` Gap Analysis

Use this section when choosing the next optimization task. The `stella_vslam` baselines below are the fixed head-250 numbers from `eval/stella_comparison_results.md`. The SimpleVisualSLAM numbers are either the current gates or the best retained post-improvement numbers called out above.

General rule for all experiments:

- change **one knob at a time**
- rerun the narrowest relevant gate first
- only after a gate improves, rerun the more expensive comparison benchmark

### 10.1 `xyz_depth` gap: about `1.24x`

- SimpleVisualSLAM: `0.011042 m`
- `stella_vslam`: `0.00889256 m`
- Loss profile: small local pose noise, not catastrophic drift

Ordered experiments:

1. `src/backend/local_mapping.cc:394-441`
   - Change:
     - `getBestCovisibilityKeyframes(15)` -> `getBestCovisibilityKeyframes(20)`
     - `max_ba_landmarks = 800` -> `1200`
   - Expected effect:
     - tighter local geometry on short RGB-D clips
     - likely small but real ATE reduction with low algorithmic risk

2. `src/backend/optimizer.cc:192-199`
   - Change:
     - metric depth prior sigma `0.02` -> `0.015`
   - Expected effect:
     - stronger trust in true metric depth during BA
     - should reduce z-axis pose jitter in depth-backed runs

3. `src/tracking/tracking.cc:1041-1119`
   - Change:
     - `kMinBootstrapCorrespondences 30 -> 40`
     - fallback pose gate `55 -> 45` and `180 -> 120`
   - Expected effect:
     - fewer noisy bootstrap PnP solves in already-well-constrained RGB-D cases
     - likely low single-digit percent gain, but safe

### 10.2 `xyz_mono` gap: about `2.0x` on the current gate

- Best retained SimpleVisualSLAM: about `0.036 m` after `c5bcbe1`
- Current strict gate: `0.028136 m`
- `stella_vslam`: `0.01413570 m`
- The hard regression blocker is resolved, but the pass margin is thin

Most likely causes:

- mono initializer still picks homography too often on non-planar motion
- mono keyframe cadence is still too sparse early in the run
- local BA window is too small for early scale/orientation stabilization

Ordered experiments:

1. `src/tracking/initializer.cc:109-129`
   - Change:
     - `use_H = ratio > 0.50f` -> `use_H = ratio > 0.60f`
   - Expected effect:
     - bias initialization toward the fundamental/essential path on `xyz_mono`
     - should reduce bad early scale/translation estimates
   - Why first:
     - minimal code churn
     - directly targets the improvement area that already moved `xyz_mono` once

2. `src/tracking/initializer.cc:12-17`, `src/tracking/initializer.cc:181-253`
   - Change:
     - `kInitMedianParallaxDeg 0.8 -> 0.6`
     - `kInitMinTriangulatedPoints 70 -> 60`
   - Expected effect:
     - make initialization survive slightly more forward-motion cases
     - may reduce outright bad starts at the cost of higher false-init risk
   - Guardrail:
     - rerun `xyz_mono_head_repro` immediately after this change; do not stack more changes first

3. `src/tracking/tracking.cc:791-824`
   - Change:
     - keep current behavior for depth runs, but for pure mono:
       - `max_frames_since_last_kf 12 -> 8`
       - low tracked threshold `60 -> 80`
       - tracked/reference ratio trigger `0.65 -> 0.75`
   - Expected effect:
     - denser early keyframes
     - better propagation support and lower drift

4. `apps/run_mono.cc:311`
   - Change:
     - `cv::ORB::create(2000)` -> `cv::ORB::create(2500)` or `3000`
   - Expected effect:
     - more frame-to-frame matches and more 3D-2D propagation in mono
     - extra CPU cost, but easy to measure

5. `src/backend/local_mapping.cc:394-441`
   - Change:
     - local BA neighbor window `15 -> 20` or `25`
   - Expected effect:
     - stronger early local optimization
     - helps both `xyz_mono` and `room_mono`

### 10.3 `room_depth` gap: about `3.3x` in the best retained state

- Best retained SimpleVisualSLAM: `0.0695 m`
- Current strict repro gate: `0.079914 m`
- `stella_vslam`: `0.02110508 m`

Most likely causes:

- pose graph is still too weak compared to a mature essential-graph implementation
- local BA window remains small before loop correction
- current loop-edge weighting improves safety, but correction strength is still limited

Ordered experiments:

1. `src/backend/optimizer.cc:647-714`
   - Change:
     - add explicit sequential odometry edges between consecutive keyframes
   - Expected effect:
     - stronger pose-graph backbone through room revisits
     - likely the highest-impact missing structural piece
   - Reason:
     - current graph is built from snapshot covisibility edges plus loop edges only

2. `src/backend/local_mapping.cc:394-441`
   - Change:
     - local BA neighbors `15 -> 25`
     - `max_ba_landmarks 800 -> 1200`
   - Expected effect:
     - cleaner geometry before any loop closure
     - should reduce the burden on the pose graph

3. `src/loop_closing/loop_closing.cc:811-891`
   - Change:
     - metric/sensor-depth loop weight from `3.0 + 4.0 * confidence` to `4.0 + 6.0 * confidence`
   - Expected effect:
     - stronger trusted loop constraints
     - more decisive correction when the match is already good

4. `src/backend/optimizer.cc:824-855`
   - Change:
     - IRLS cutoff `median + 2.5 * MAD` -> `median + 2.0 * MAD`
   - Expected effect:
     - stale/bad loop edges get downweighted sooner
     - more helpful on long room runs than on short xyz runs

5. `src/loop_closing/loop_closing.h:124-125`
   - Change:
     - `max_loop_candidates_ 4 -> 8`
     - `min_loop_score_ 0.01 -> 0.005`
   - Expected effect:
     - broader candidate search when BoW/local search is too conservative
   - Risk:
     - can increase false positives, so do this only after checking logs for candidate starvation

### 10.4 `room_mono` gap: about `6.5x`

- SimpleVisualSLAM current gate: `0.176506 m` (improved from `0.197374 m` after the pre-VIO mono-room work)
- `stella_vslam`: `0.02743546 m`
- Published loop-enabled comparison saw **no loop detections inside the first 250 frames**

Most likely causes:

- front-end support thins out too early
- keyframe cadence is too conservative for pure mono in a revisit-heavy room sequence
- loop search is not reaching a trustworthy closure soon enough

Ordered experiments:

1. ~~`apps/run_mono.cc`: `cv::ORB::create(2000)` → `3000`~~ **TESTED 2026-04-20 — REGRESSED, reverted.** Predicted "strongest immediate increase in mono match density" but:
   - `room_mono_head_repro`: `0.177 m` → `0.227 m` (worse)
   - `room_mono_accel_head_repro`: `0.164 m` → `0.236 m` (above `0.170 m` ceiling)
   - `room_depth_head_repro`: `0.080 m` → `0.068 m` (actually improved)
   - Mono is more sensitive to feature count at the 250-frame boundary than plan.md anticipated — more keypoints = more outlier BA candidates when the map is still thin. Do not re-try without pairing it with a depth-detecting gate bypass.

2. `src/tracking/tracking.cc:775-824`
   - Change for mono only:
     - `min_frames_since_last_kf 3 -> 2`
     - `max_frames_since_last_kf 12 -> 7`
     - `min_tracked_threshold 60 -> 80`
     - tracked/reference ratio trigger `0.65 -> 0.75`
   - Expected effect:
     - denser room map
     - more stable anchors through revisits

3. `src/backend/local_mapping.cc:196`, `src/backend/local_mapping.cc:394-441`
   - Change:
     - triangulation neighbors `15 -> 25`
     - local BA neighbors `15 -> 25`
     - `max_ba_landmarks 800 -> 1500`
     - BA iterations `15 -> 20`
   - Expected effect:
     - better local map quality and more cross-view constraints on room mono

4. `src/tracking/initializer.cc:109-129`
   - Change:
     - same H/F decision change as `xyz_mono` (`0.50 -> 0.60`)
   - Expected effect:
     - room mono quality is dominated by early bad starts even more than xyz mono

5. `src/loop_closing/loop_closing.h:123-133`, `src/loop_closing/loop_closing.cc:471-753`
   - Change for mono experiments:
     - `max_loop_candidates_ 4 -> 8`
     - `min_matches_for_loop 50 -> 40`
     - `min_loop_inliers_ 30 -> 25`
     - compensate by tightening `max_sim3_residual_ 0.25 -> 0.20`
   - Expected effect:
     - more room-mono loop attempts without blindly accepting weak geometry

6. `src/tracking/tracking.cc:1041-1119`
   - Change:
     - when falling back to all landmarks, reduce `fallback_gate_px 180 -> 120`
     - require at least `40` correspondences before running fallback PnP
   - Expected effect:
     - fewer catastrophic long-range false PnP corrections on sparse room frames

### 10.5 Ceres Changes Most Likely To Help

These are concrete solver/backend changes worth trying after the first threshold-only ablations:

1. `src/backend/optimizer.cc:520-529`
   - current local BA always uses `DENSE_SCHUR`
   - try switching to `SPARSE_SCHUR` when the BA window gets larger than roughly `400` landmarks or `15` keyframes
   - expected effect:
     - makes 20-25 keyframe local BA windows practical

2. `src/backend/optimizer.cc:782-785`
   - pose graph already uses `SPARSE_NORMAL_CHOLESKY` + `SUITE_SPARSE`
   - if sequential edges are added, raise loop-present iteration cap `90 -> 120`
   - expected effect:
     - more complete convergence after a real loop constraint

3. `src/backend/optimizer.cc:752-760`
   - if scale drift persists in metric-depth mode, raise `kMetricScalePriorWeight 100 -> 300`
   - expected effect:
     - keep the pose graph closer to true metric scale

### 10.6 Algorithmic Changes Most Likely To Help

If threshold changes plateau, these are the next non-trivial algorithmic moves:

1. Add an **essential-graph backbone** in pose graph optimization.
   - sequential edges + parent/spanning-tree edges + strong covisibility + loop edges
   - current implementation is closer to "snapshot covisibility graph + loop edges"

2. Replace brute-force descriptor scans with **search-by-projection + orientation consistency** in the front end.
   - most relevant code: `src/tracking/tracking.cc:1323-1492` and `src/tracking/tracking.cc:949-1119`

3. Improve map-point culling by observation count / parallax / viewing angle before BA.
   - current culling is modest; better map hygiene should help both room scenarios

4. If room-mono loop search still never fires, add a **candidate persistence rule** before `computeSim3()`.
   - accept a candidate only if it stays the top candidate for multiple nearby keyframes
   - this is safer than simply lowering all thresholds globally

---

## 11. Priority

Ordered for **2026-04-21 → next session**. Do the practical items first, then deep-dive:

1. **Land PR #3.**
   - Branch: `vio-integration`, head `8e6f4e7`, not yet merged. 77/77 tests, 7/7 gates, EuRoC cross-seq validated.
   - After merge, bump local `master`, rerun `ctest` + `scripts/check_regression_gate.py --all-gates` as a smoke check, delete the local branch.

2. **Diagnose MH_03 regression (§8.11 / §16.4).**
   - 5-way IMU-path bisection: (a) accel_buffer_ population only, (b) gravity alignment only, (c) gravity prior only, (d) `predictVelocityFromImu` only, (e) `reconcileVelocityWithVisual` only. Binary-search which sub-path introduces the +0.4 m vs visual-only.
   - Success criterion: MH_03 mean ATE at or below visual-only `2.48 m` with `--accel` on.

3. **Commit a proper `eval/` artifact for the EuRoC cross-seq sweep.**
   - Mirror `eval/room_depth_600frame_report.md`. Include sequence, frames, mean/median/max ATE, the exact `run_mono` command, the `evo_ape` command, and the commit hash.
   - Same-day follow-up: a machine-readable `eval/euroc_vio_sweep.json` that `scripts/build_leaderboard.py` can ingest.

4. **Probe `plan.md` §10 experiments one at a time.**
   - §10.1 (xyz_depth) is the smallest gap and the lowest risk — start there.
   - Rule: narrowest gate first, all-gates second, revert on any ceiling breach.
   - Update the §10 table entry (✅ kept / ❌ reverted + measured numbers) inside the same commit.

5. **Phase D deep-dives.** Choose one; do not stack.
   - **ORB-SLAM3-style MAP VI init (§16.5):** non-linear joint optimization over {scale, gravity, velocities, accel_bias, gyro_bias} on the first ~30 KFs. Goal: make MH_01 accept VI init with rot_rms ≤ 0.08 rad.
   - **EuRoC stereo rectification fix (§8.13):** proper fisheye / radtan support so `--stereo` on EuRoC isn't a regression vs mono.
   - **ROS2 VIO hookup:** wire `Tracking::imu_buffer_` / `setImuToCameraExtrinsic` through `ros2/src/slam_node.cc`.

Previously-priority items now complete:

- ~~`xyz_mono` recovery~~ (done, `0.028 m` gate passing)
- ~~`room_depth` pose graph~~ (done 2026-04-15)
- ~~`eval/stella_comparison_results.md` fair head-250 refresh~~ (done)
- ~~600-frame loop report~~ (`eval/room_depth_600frame_report.md`)
- ~~Real EuRoC verification~~ (done 2026-04-19 / 2026-04-20 via Wayback; 5 sequences on disk, see §2.7–§2.8)
- ~~VIO Stage 0b / 0c~~ (landed on `master`; PR #3 pending)

---

## 12. AI Agent Instructions

1. Read in this order:
   - **this file** (especially §2.1, §2.4, §2.7, §2.8, §8, §11, §16)
   - `apps/run_mono.cc`
   - `src/tracking/tracking.cc` + `tracking.h`
   - `src/tracking/initializer.cc`
   - `src/tracking/visual_inertial_initializer.{h,cc}` (if touching the VIO path)
   - `src/backend/local_mapping.cc`
   - `src/backend/optimizer.{h,cc}` (the residual definitions live in the header)
   - `src/sensors/imu_preintegrator.h` + `imu_preintegration_span.h`
   - `src/loop_closing/loop_closing.cc`

2. Use the narrowest benchmark that can validate your change:
   - **mono front-end**: `xyz_mono_head_repro` *first* (thin margin), then `room_mono_head_repro`, then `room_mono_accel_head_repro` (thinnest margin — 4 %). Only after all three pass, run `--all-gates`.
   - **pose-graph / loop**: `room_depth_head_repro`, then the loop-enabled comparison preset
   - **metric-depth**: rerun the metric-depth scripts on `xyz` before touching `room`
   - **VIO**: EuRoC mono `--accel` on MH_01 *first* (best-behaved, biggest gain), then cross-check on V1_01 (VI init accepts) and MH_03 (open regression; at least should not regress more). Visual-only baseline on the same sequence is a pre-requisite: run it too.

3. For threaded changes, test both modes:
   - `--repro-eval`
   - normal async mode with loop closing enabled

4. Keep the repo honest:
   - if you change a number that appears in `eval/*.md` or `README.md`, rerun the benchmark and update the artifact
   - do not propagate retained-note numbers into public docs without a fresh rerun

5. Do not silently change protocol:
   - keep `evo_ape` flags aligned with `eval/regression_baselines.json` for TUM gates (`--align --correct_scale --t_max_diff 0.05`)
   - use the same flags for EuRoC mono (Sim3 is mandatory — mono scale is unobservable)
   - do not redefine what a regression gate means without updating the harness/docs together

6. Do not commit unless explicitly asked.
   - If a human later asks for a commit, keep one logical change per commit and include the exact validation command(s).
   - **Never add AI-generated markers** ("Generated with ...", "Co-Authored-By: Claude", etc.) to commit messages or PR descriptions — this project forbids them.

7. When probing `plan.md` §10 experiments, trust the *direction* but not the *numbers*:
   - The §10 lists were written against an earlier codebase state. Predicted gains have flipped sign at least once (ORB 3000 — see §8.12).
   - Rule: narrowest gate first, all-gates second, revert on any ceiling breach, update the §10 table entry in the same commit with the measured delta and the date.

8. When touching VIO:
   - Default sigmas (§7.8) were tuned empirically on EuRoC. Do not tighten `SVSLAM_BA_VELOCITY_PRIOR_SIGMA_M` below `0.2` without fresh evidence on MH_01.
   - The gyro-bias first-order Jacobian inside the BA rotation residual was tried and reverted (§8.10). Do not re-enable without a working VI init first.
   - `plan.md` §16 is the VIO-specific handoff; read it before making any change that touches `ImuPreintegrationSpan`, `VelocityPreintegrationError`, or the `VisualInertialInitializer`.

---

## 13. Build And Run Commands

### 13.1 Build / Test

```bash
cmake -S . -B build_codex -G Ninja -DBUILD_TESTS=ON -DUSE_DEPTH_DL=ON
cmake --build build_codex -j$(nproc)
ctest --test-dir build_codex --output-on-failure
./build_codex/run_mono --version
./build_codex/run_mono --help
```

### 13.2 Regression Gates

```bash
python3 -u scripts/check_regression_gate.py --build build_codex --gate xyz_mono_head_repro --quiet
python3 -u scripts/check_regression_gate.py --build build_codex --gate xyz_depth_head_repro --quiet
python3 -u scripts/check_regression_gate.py --build build_codex --gate room_mono_head_repro --quiet
python3 -u scripts/check_regression_gate.py --build build_codex --gate room_depth_head_repro --quiet
python3 -u scripts/check_regression_gate.py --build build_codex --gate room_depth_accel_head_repro --quiet
```

### 13.3 Comparison / Benchmark Presets

```bash
BUILD=build_codex bash scripts/verify_comparison_benchmark.sh xyz_depth
BUILD=build_codex bash scripts/verify_comparison_benchmark.sh xyz_mono
BUILD=build_codex bash scripts/verify_comparison_benchmark.sh room_depth
BUILD=build_codex bash scripts/verify_comparison_benchmark.sh room_mono
python3 scripts/build_leaderboard.py --build build_codex --quiet
```

### 13.4 TUM Runs

```bash
./build_codex/run_mono --tum data/tum/rgbd_dataset_freiburg1_xyz --depth --max-frames 250 --repro-eval --no-viz
./build_codex/run_mono --tum data/tum/rgbd_dataset_freiburg1_xyz --max-frames 250 --repro-eval --no-viz
./build_codex/run_mono --tum data/tum/rgbd_dataset_freiburg1_room --depth --max-frames 250 --no-viz data/ORBvoc.txt
./build_codex/run_mono --tum data/tum/rgbd_dataset_freiburg1_room --depth --max-frames 600 --no-viz data/ORBvoc.txt
```

### 13.5 Metric Depth

```bash
./build_codex/run_mono --tum data/tum/rgbd_dataset_freiburg1_xyz \
  --metric-depth-model models/depth_anything_v2_metric_indoor_small.onnx \
  --max-frames 250 --no-viz

python3 scripts/print_ate_mean.py \
  data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt trajectory.txt
```

### 13.6 EuRoC Mono / Stereo

```bash
./build_codex/run_mono \
  --euroc data/euroc/test_seq \
  --euroc-camera-config config/examples/euroc_mh01.json \
  --max-frames 100 \
  --no-viz \
  --run-summary-json eval/euroc_mono_summary.json

./build_codex/run_mono \
  --euroc data/euroc/test_seq \
  --euroc-camera-config config/examples/euroc_mh01.json \
  --stereo \
  --max-frames 100 \
  --no-viz \
  --run-summary-json eval/euroc_stereo_summary.json
```

---

## 14. stella_vslam Battle Log (2026-04-13〜14)

Every parameter change tried to close the gap, with measured results:

### xyz_depth (target: < 0.00889, stella baseline)

| Change | ATE | vs baseline | Verdict |
|--------|-----|-------------|---------|
| Baseline (before tuning) | 0.01214 | — | — |
| depth sigma 0.02→0.015 | 0.01176 | -3% | **kept** |
| Huber loss 1.0→0.5 | 0.01321 | +9% | reverted |
| search radius 100→80 | 0.01207 | -0.6% | reverted (negligible) |
| covis KF 15→20 | 0.01111 | -8% | **kept** |
| sigma 0.015 + covis 20 + BA iter 20 | **0.01104** | **-9%** | **committed (ef8e04e)** |

Best achieved: **0.01104** — still 1.24x behind stella (0.00889). The gap is likely structural.

### xyz_mono

| Change | ATE | Verdict |
|--------|-----|---------|
| Lowe ratio 0.75→0.70 | 0.0413 | **broke gate** (ceiling 0.030) |
| Lowe ratio reverted to 0.75 | 0.0223 | **recovered (f0dd425)** |

### Cross-scenario effects

| Change | xyz_depth | xyz_mono | room_depth | room_mono |
|--------|-----------|----------|------------|-----------|
| Lowe 0.70 | neutral | **broke** | neutral | improved |
| Covis 20 + sigma 0.015 | improved | neutral | neutral | neutral |

### Lessons learned

1. **Lowe ratio is globally sensitive** — 0.70 helps room_mono but destroys xyz_mono. Keep at 0.75.
2. **Depth sigma helps only depth scenarios** — 0.015 is better than 0.02 for sensor depth.
3. **Covis window 20 is safe** — unlike the earlier attempt (before IRLS), it now passes all gates.
4. **Huber tightening hurts** — 0.5 is too aggressive, outlier rejection becomes noise rejection.
5. **The xyz_depth gap (1.24x) may be structural** — Ceres dense BA vs g2o sparse Schur, map management differences.

### What the next agent should try (ordered by expected impact)

**For xyz_depth (0.01104 → target < 0.00889):**
1. `src/backend/optimizer.cc`: Try adaptive Huber (start at 1.0, reduce to 0.7 after 5 iterations)
2. `src/tracking/tracking.cc:trackLocalMap()`: Add depth prior weighting in PnP — currently PnP treats all points equally, but depth-backed points should be more trusted
3. `src/tracking/tracking.cc:trackLocalMap()`: Try 3-pass PnP (current: 2-pass) with progressively tighter reproj threshold (8px → 5px → 3px)
4. `src/backend/optimizer.cc:bundleAdjustment()`: Try SPARSE_SCHUR solver for local BA (currently uses DENSE_SCHUR)

**For room_depth (head-250 ~`0.08 m` → target < 0.02):**
1. 600-frame loop-enabled sanity: **`eval/room_depth_600frame_report.md`** (mean ATE ~`0.110 m`, 2026-04-15)
2. `src/loop_closing/loop_closing.cc`: Improve Sim3 estimation quality — current RANSAC only does 200 iterations
3. `src/backend/optimizer.cc:poseGraphOptimization()`: The IRLS 2-pass is in, but try 3-pass with tighter Cauchy threshold (sequential KF edges are already in)

**For room_mono (0.2688 → target < 0.027):**
1. This is 9.8x behind and likely requires fundamentally better mono map maintenance
2. Focus on landmark lifecycle: creation, culling, and observation propagation
3. Consider more aggressive keyframe insertion for mono (every 10 frames instead of 20)

### What NOT to do

- Don't change Lowe ratio below 0.75 (breaks xyz_mono, verified)
- Don't increase covis KF above 20 without checking all 5 gates
- Don't add g2o (GPL license conflict with BSD-2-Clause)
- Don't claim stella_vslam is beaten until reproduced 3x with identical protocol
- Don't modify regression gate semantics without updating baselines

## 15. Mono Room Handoff (2026-04-15) — historical

> **Historical note (2026-04-21):** this section was written when `room_mono_head_repro` was the single active frontier and VIO work had not started. The §15.1 safe-state number (`0.197374 m`) has since moved to `0.176506 m` on the current master. The guidance inside this section (favor ranking/selection over threshold widening; frame-199 diagnostic; trace-only diffs before behavior changes) is still valid for any mono-front-end change. Treat everything below as design context, not a current TODO — §16 supersedes it for the VIO frontier, and §11 / §10.4 carry the current mono TODO.

This section is the immediate handoff for the next agent. It is intentionally more concrete than the older battle log above because the active frontier is now very narrow: `room_mono_head_repro` is the only materially weak gate left in the head-250 harness, and repeated broad threshold sweeps have already been ruled out.

### 15.1 Current Safe State

The retained `build_codex` branch is stable and reproducible.

- `room_mono_head_repro = 0.197374 m`
- `xyz_mono_head_repro = 0.028136 m`
- `room_depth_head_repro = 0.079914 m`
- `room_depth_accel_head_repro = 0.057702 m`
- `xyz_depth_head_repro = 0.011042 m`
- `ctest --test-dir build_codex --output-on-failure = 58/58 PASS`
- `python3 -u scripts/check_regression_gate.py --build build_codex --gate room_mono_head_repro --quiet`
  - SHA-256: `2bf026342e6fd964…` (matches `--all-gates` run on 2026-04-16)
  - mean ATE: `0.197374 m`

The most important retained mono-side code changes are:

- `src/tracking/initializer.cc`
  - homography / fundamental selection moved from `ratio > 0.50f` to `0.60f`
- `src/tracking/tracking.cc`
  - keyframe decision trace support
  - sparse mono keyframe ratio tightening for the early `50 <= frame_id < 80` zone
  - bootstrap / pose-filter instrumentation in `trackLocalMap()`
  - late sparse mono relaxed pose-filter retry in fallback matching
  - relocalization candidate scoring improvements, especially local-candidate prioritization and "best successful candidate" selection instead of first success
- `src/core/heuristic_reference_keyframe_policy.cc`
  - late sparse mono refresh rule
- `src/backend/local_mapping.cc`
  - skip the all-keyframe triangulation fallback when the current mono keyframe is sparse and unsupported
- `src/backend/optimizer.cc`
  - optional sequential (time-adjacent) pose-graph edges when covisibility is sparse
- `apps/run_mono.cc` / `src/tracking/tracking.h`
  - `--keyframe-trace-csv` plumbing; `RoomFocusTrace` for mono frames 100–125

This safe state should be treated as the baseline. Any mono-front-end experiment should return to this exact state before the next ablation.

### 15.2 What Already Improved `room_mono`

The important recent milestones were:

1. `0.336660 -> 0.235966`
   - achieved by adding a narrow sparse-mono keyframe ratio rule in the early room segment
2. `0.235966 -> 0.234953`
   - achieved by allowing late sparse mono reference refresh in the heuristic reference policy
3. `0.234953 -> 0.228579`
   - achieved by stopping unsupported sparse mono keyframes from falling back to all-keyframe triangulation in local mapping
4. `0.228579 -> 0.228355`
   - achieved by adding a narrow late-sparse fallback pose-filter retry in `trackLocalMap()`
5. `0.228355 -> 0.220497`
   - achieved by improving `relocalize()` candidate ordering:
     - prefer stronger local candidates
     - evaluate all successful candidates
     - keep the best success instead of accepting the first success
     - use pose-quality tie-break only in near-ties
6. `0.220497 -> 0.197374`
   - achieved by widening the mono keyframe spacing floor (`min_frames_since_last_kf` 4 vs 3 for RGB-D) and tightening mono bootstrap visible-pool floor / instrumentation (`BootstrapStats`, `RoomFocusTrace`)

This matters because the biggest remaining gain came from relocalization quality, not from broad keyframe-cadence changes; the latest step is a small cadence + bootstrap guardrail pass that also moved `room_mono` under `0.20 m` on the head-250 harness.

### 15.3 What Has Been Ruled Out

Many obvious knobs have already been tried and should not be re-tried blindly:

- ORB count increase `2000 -> 3000`
  - hurt `xyz_mono`, no durable `room_mono` gain
- globally lowering Lowe ratio below `0.75`
  - helps some room-mono behavior but breaks `xyz_mono`
- more aggressive mono keyframe insertion or cadence changes applied broadly
  - often helps room locally, but repeatedly pushes `xyz_mono` over the `0.030000 m` ceiling
- widening local BA windows
  - degraded room
- broad relocalization quality-first selection
  - worse than current near-tie-only logic
- direct rescue of frame-199-style near-miss PnP by lowering inlier threshold
  - increases apparent recovery but worsens final ATE
- large-jump relocalization suppression after recovery
  - also worse
- multiple broad sparse-mono policy relaxations in reference selection
  - either no-op or harmful

More specifically, the following late-room bootstrap ideas were tested and reverted:

- all-landmark bootstrap subset rescue
  - no retained gain
- visible-only bootstrap broadening by lowering the visible-pool floor from `80 -> 70`
  - exact no-op at trajectory level on its own
- visible-only bootstrap descriptor relaxation (`dist 65 -> 70`, `ratio 0.75 -> 0.80`) combined with the `70` floor
  - frame 199 improved locally, but room ATE regressed badly to `0.289405 m`

The last point is especially important: increasing correspondence count around the frame-199 collapse is not sufficient. Some of the added matches are low-quality and poison the downstream pose.

### 15.4 Best Current Diagnosis Of `room_mono`

The current failure mode is concentrated in the late-room segment around frames `199-202`.

Facts established from traces:

- `room_mono` exhibits repeated late relocalization and sparse keyframe churn that `xyz_mono` does not
- the critical bad zone is not the early mono initialization anymore; that side is already stabilized enough to keep `xyz_mono` under ceiling
- frame `199` is the most informative single frame for diagnosis

From `log/room_mono_trace_20260415ab.log` on the safe state:

- frame 199 local-map source composition:
  - `ref=97`
  - `ref_neighbors=583`
  - `prev_ref=0`
  - `prev_neighbors=0`
  - `global=0`
- visibility breakdown:
  - total landmarks to check: `680`
  - visible: `76`
  - out-of-bounds: `575`
  - reference bucket contributes `47` visible / `7` direct matches
  - reference-neighbor bucket contributes only `29` visible / `5` direct matches

This means the old intuition "frame 199 fails because the map falls back to the whole map" was wrong. The bad behavior is already present inside the local map:

- most of the neighbor landmarks are simply off-screen
- the visible pool is small but non-trivial (`76`)
- broadening from the visible pool to all `680` local landmarks was suspected to be harmful
- however, removing that broadening alone did **not** change the final trajectory

That no-op result is a strong clue. It implies that pool width by itself is not the dominant variable. The next lever is candidate quality and ranking inside fallback bootstrap, not just how many landmarks are offered.

### 15.5 Critical Logs And What They Proved

The most useful recent logs are:

- `log/room_mono_trace_20260415ab.log`
  - safe-state local-map source analysis
  - established that frame 199 is dominated by out-of-bounds local-neighbor landmarks
- `log/room_mono_trace_20260415ac.log`
  - verified that visible-pool floor `80 -> 70` makes frame 199 stay on the visible pool (`from_all=0`)
  - trajectory SHA remained the same safe SHA, so that change alone was effectively a no-op
- `log/room_mono_trace_20260415ae.log`
  - visible-pool floor `70` plus visible-only descriptor relaxation
  - frame 199 improved numerically:
    - `candidates 3 -> 7`
    - `added_post_pose 2 -> 5`
    - `from_all=0`
  - but the gate degraded to `0.289405 m`

The most recent failed experiment is worth stating explicitly:

- safe state:
  - frame 199 bootstrap: `pool=76`, `candidates=3`, `added_post_pose=2`, `from_all=0` only if floor is lowered, otherwise broadens to all-local pool
  - final safe ATE: `0.197374 m`
- failed variant:
  - frame 199 bootstrap: `pool=76`, `candidates=7`, `added_post_pose=5`, `from_all=0`, relaxed visible-only descriptor gates
  - final room ATE: `0.289405 m`

Conclusion: the missing ingredient is not "more correspondences". The missing ingredient is "better correspondences".

### 15.6 Most Likely Next Direction

The next agent should not spend another cycle on broad threshold sweeps. The best next hypothesis is:

> In late sparse mono fallback bootstrap, the additional correspondences should be ranked by a quality score that includes geometric consistency, not just descriptor distance.

Concretely, the most promising next step is inside `src/tracking/tracking.cc::trackLocalMap()`:

1. Keep the safe-state behavior first.
2. Add a trace-only branch that records the top fallback candidates before they are appended:
   - descriptor distance
   - Lowe ratio margin
   - octave
   - landmark source bucket (`ref` vs `ref_neighbors`)
   - current-pose reprojection error estimate before the fallback pose filter
   - projected depth
3. For late sparse mono only, compare two candidate orderings:
   - current: descriptor distance only
   - proposed: rank by "passes coarse reprojection sanity" first, then descriptor distance
4. Keep the acceptance thresholds unchanged on the first try.
   - Only change ordering.
   - Do not widen `dist` or `ratio` first.

The core idea is to improve the quality of the first `kMaxBootstrapMatches` correspondences rather than increasing the raw count.

### 15.7 Concrete Suggested Experiment Order

If Copilot or another agent resumes from here, the safest experiment order is:

1. Add trace for fallback candidate ranking quality in `trackLocalMap()`
   - no behavior change
   - compare frame `199` and the two following relocalization frames
2. Try a ranking-only experiment for late sparse mono visible-pool bootstrap:
   - sort candidates by a coarse reprojection-consistency bucket, then descriptor distance
   - keep existing distance / ratio thresholds
3. If that helps `room_mono`, immediately run:
   - `python3 -u scripts/check_regression_gate.py --build build_codex --gate room_mono_head_repro --quiet`
   - `python3 -u scripts/check_regression_gate.py --build build_codex --gate xyz_mono_head_repro --quiet`
   - `ctest --test-dir build_codex --output-on-failure`
4. If ranking-only is a no-op, consider source-aware ranking:
   - prefer `ref` landmarks over `ref_neighbors` when the score is close
   - frame 199 currently gets only `29` visible neighbor landmarks, and many neighbor landmarks are off-screen
5. Only after ranking experiments fail should anyone revisit threshold changes

### 15.8 Practical Rules For The Next Agent

- Treat `0.197374 m` and SHA `2bf026342e6fd964…` as the safe-state anchor.
- Do not stack multiple mono changes before rerunning the two mono gates.
- Rerun `xyz_mono_head_repro` immediately after any room-mono improvement; its pass margin is still thin.
- Prefer small ranking / selection changes over threshold widening.
- If a change improves frame 199 local stats but worsens final ATE, revert it quickly; this happened already.
- Keep trace output additions if they are useful and low-risk, but avoid shipping broad behavior changes without gate confirmation.

### 15.9 Minimal Commands To Resume

Safe-state verification:

```bash
python3 -u scripts/check_regression_gate.py --build build_codex --gate room_mono_head_repro --quiet
python3 -u scripts/check_regression_gate.py --build build_codex --gate xyz_mono_head_repro --quiet
ctest --test-dir build_codex --output-on-failure
```

Late-room tracing:

```bash
./build_codex/run_mono \
  --tum data/tum/rgbd_dataset_freiburg1_room \
  --max-frames 250 \
  --repro-eval \
  --no-viz \
  --reference-policy heuristic \
  > log/room_mono_trace_next.log 2>&1
```

Frame-199 inspection:

```bash
sed -n '3928,3955p' log/room_mono_trace_next.log
rg -n "BootstrapStats|Relocalize: Candidate KF|Relocalize: Matched with KF" log/room_mono_trace_next.log
```

## 16. VIO Handoff (2026-04-21)

This section supersedes §15 for any change that touches the IMU path. §15 remains the right handoff for mono-only front-end work.

### 16.1 Current Safe State

Master: `46a726d`. PR #3 (branch `vio-integration`, head `8e6f4e7`) adds a flaky-test fix and a gravity-prior env knob.

Build + tests:

```bash
cmake -S . -B build -G Ninja -DBUILD_TESTS=ON
cmake --build build -j$(nproc) --target svslam_core svslam_tests run_mono
ctest --test-dir build --output-on-failure   # 77 / 77 PASS
python3 scripts/check_regression_gate.py --build $PWD/build --data-tum $PWD/data/tum --all-gates --quiet
# → 7 / 7 PASS, bitwise-identical trajectories
```

EuRoC mono `--accel` baselines (reproduce before changing anything):

```bash
./build/run_mono --euroc datasets/euroc/MH_01_easy --accel --reference-policy heuristic \
  --skip-frames 0 --no-viz --repro-eval
evo_ape tum datasets/euroc/MH_01_easy/gt_tum.txt build/trajectory.txt \
  --align --correct_scale --t_max_diff 0.05 --no_warnings
# → mean ~1.41 m
```

Matching visual-only baseline (same command without `--accel`) → `3.44 m`.

### 16.2 Pipeline Diagram

```
Frame N (mono image) ───┐
                        │
                        ▼
              Tracking::addFrame
                        │
                        ▼
             Tracking::track
              ├─ motion-model pose predict
              ├─ stationary detection (accel_buffer_)
              ├─ predictVelocityFromImu    ──► Frame::velocity_  (IMU body, world)
              ├─ trackReferenceKeyframe  (reprojection / PnP)
              ├─ trackLocalMap
              ├─ reconcileVelocityWithVisual ──► blended Frame::velocity_
              └─ if needNewKeyframe:
                   Keyframe::Keyframe(Frame)        ─► copies velocity_, biases
                   setKeyframeGravity                (gravity in camera frame)
                   populateKeyframeImuSpan           ─► ImuPreintegrationSpan (delta_R/v/p, dt, T_cam_imu, from_kf_id)
                   tryVisualInertialInit             (if enough KFs and vi_init_done_ == false)
                        └─ VisualInertialInitializer::initialize
                             ├─ Stage 1: closed-form gyro bias (capped at 0.05 rad/s)
                             └─ Stage 2: LSQ for {scale, gravity, velocities}
                        └─ on success: rescale map, rotate world to gravity-Z,
                                        applyGyroBiasCorrectionToSpans (first-order Forster),
                                        Optimizer::setPreintegrationResidualEnabled(true)

LocalMapping::optimization    ─► Optimizer::bundleAdjustment
   ├─ ReprojectionError (per landmark observation)
   ├─ DepthPriorError        (if depth available)
   ├─ GravityPriorError      (per KF with has_gravity_, env-tunable weight)
   ├─ VelocityPreintegrationError  (9-DoF, when prev_imu_span_ valid)
   ├─ VelocityDeltaPriorError       (loose fallback)
   ├─ BiasAnchorError * 2            (accel + gyro anchors)
   └─ BiasRandomWalkError * 2       (between consecutive KFs)
```

### 16.3 Key Invariants

- `Frame::velocity_` is world-frame velocity of the **IMU body**, not the camera. The lever-arm term `t_bc` is baked into `VelocityPreintegrationError` so BA reconciles properly.
- `ImuPreintegrationSpan::T_cam_imu` is captured at span-creation time, not looked up at BA time. If the EuRoC extrinsic ever becomes dynamic, revisit this.
- Rotation alignment matters: the `initializeWithDepth` / `initialize` paths now rotate the gravity estimate from the IMU frame into the camera frame before building `T_align`. This was a silent bug for TUM (IMU ≈ camera) but catastrophic on EuRoC before the fix landed in `bebc7d5`.
- Default: preintegration residual is **always on** when `has_velocity_` is set. VI init's role is to refine scale / gravity / velocities + apply the first-order gyro correction to spans, *not* to gate the residual. The old gated behavior is available via `SVSLAM_VIO_GATE_PREINT=1` but has been shown to regress MH_01 catastrophically when VI init rejects (1.41 → 3.46 m).

### 16.4 Open Regression — MH_03_medium

Visual-only `2.48 m`, +VIO `2.91 m` (+17 %). The `--accel` path clearly hurts, but none of the individual components fully explains it:

| Config                                                 | Mean ATE |
| ---                                                    | ---:     |
| Visual-only                                            | `2.48 m` |
| +IMU (full default config)                             | `2.91 m` |
| +IMU, `SVSLAM_BA_VELOCITY_PRIOR_SIGMA_M=0` (preint off) | `3.04 m` |
| +IMU, `SVSLAM_BA_GRAVITY_PRIOR_WEIGHT=0` (gravity off) | `2.90 m` |

Lost-tracking events: 82 visual-only, 103–126 with `--accel`. Something in the IMU path is destabilizing Tracking itself, not just adding bad BA residuals.

Next bisection to try (one at a time, commit env knobs as needed):

1. Run with `--accel` but short-circuit `predictVelocityFromImu` to a no-op. If MH_03 recovers, the initial velocity estimate is the culprit.
2. Short-circuit `setKeyframeGravity` (skip per-KF gravity direction). Eliminates noisy per-KF gravity propagation without touching the init-time alignment.
3. Short-circuit `reconcileVelocityWithVisual`'s IMU path. Isolates the blended-velocity path from the raw-IMU path.
4. As a last step, keep only `accel_buffer_` populated (so gravity init fires once, but no per-frame IMU work). If MH_03 still regresses, the init-time `T_align` on MH_03's accel window is miscomputed.

Likely culprit on prior: short-circuit (1) or the init-time gravity alignment (4). MH_03 has aggressive flight, so a ±50 ms accel mean can drift noticeably from true gravity.

### 16.5 Stage 0c.f — Follow-ups (not in this PR)

Three candidate directions, ranked by expected payoff per hour:

1. **ORB-SLAM3-style MAP-based VI init refinement.** After the current linear two-stage solve, run a Ceres problem over the first ~30 KFs with fixed visual poses and free {scale, gravity, per-KF velocities, accel_bias, gyro_bias}. Preintegration residuals + soft bias priors. Should make MH_01 and MH_03 accept VI init by reducing the rot_rms below `0.08 rad` — the current linear gyro-bias solve is too short-window-dependent.
2. **Gyro-bias first-order Jacobian in the BA rotation residual** (re-tried after #1). The current reverted attempt regressed MH_01 2–3× because BA absorbed visual rotation noise into `bg`. Once the MAP init above delivers `bg` ≈ O(0.01 rad/s), the Jacobian becomes a refinement, not an over-fit. Gate behind `SVSLAM_BA_PREINT_ENABLE_GYRO_JACOBIAN=1` to avoid default surprises.
3. **Longer VI-init window + source ranking.** Today `SVSLAM_VIO_MIN_INIT_KEYFRAMES=15`. Accepting at 25–30 should help V1_01-class windows where the first few mono rotations are noisier. Must be paired with a `max_init_attempts` cap so we stop retrying after the window stabilizes.

### 16.6 Reference Files For Any VIO Change

Minimum set to read and understand before changing anything VIO-side:

- `src/sensors/imu.h` (ImuEntry value type)
- `src/sensors/imu_preintegrator.h` (math)
- `src/sensors/imu_preintegration_span.h` (per-KF-pair state)
- `src/core/frame.h` and `core/keyframe.h` (where velocity / biases / spans live)
- `src/tracking/tracking.h` + the VIO-named methods in `tracking.cc` (`predictVelocityFromImu`, `reconcileVelocityWithVisual`, `populateKeyframeImuSpan`, `tryVisualInertialInit`, plus the `gravity_aligned_` init paths)
- `src/tracking/visual_inertial_initializer.{h,cc}` (linear 2-stage solve)
- `src/backend/optimizer.h` — **all VIO residuals are defined here**: `VelocityPreintegrationError`, `VelocityDeltaPriorError`, `BiasAnchorError`, `BiasRandomWalkError`
- `src/backend/optimizer.cc` — the `bundleAdjustment` function plumbs them; search for "preintegration" in that file
- `apps/run_mono.cc` — EuRoC setup path, especially `setImuToCameraExtrinsic` and the preint-gate opt-in

Minimum reproducing commands are in §16.1. Once you have new numbers, update §2.8, §8, §11 in the same commit.

---

## 17. Non-Goals

Do **not** spend time on these unless a human explicitly reprioritizes them:

- restarting the reference-keyframe policy research track; `heuristic` is already the settled runtime default
- switching to `g2o` or any GPL-sensitive dependency just to chase one benchmark quickly
- rewriting the ROS2 node before the core numeric gaps are closed
- updating public benchmark claims from retained notes alone
- batching multiple unrelated threshold changes into one ablation
- changing regression-gate semantics without updating the evaluation harness and docs together
