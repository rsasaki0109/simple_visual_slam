# Fair comparison against other open-source SLAM / VO

This repo’s stated peer is **readable, BSD-friendly** systems on similar sensors—not a claim to beat every leaderboard entry on every metric.

## Why protocol matters

Published ATE numbers are **not comparable** if any of these differ:

- Trajectory length / frame window (`--skip-frames`, `--max-frames`)
- Alignment mode (`evo_ape` `--align`, `--correct_scale`, `--t_max_diff`, …)
- Monocular vs RGB-D vs stereo
- Loop closing on/off and vocabulary
- CPU / threading (Ceres `SVSLAM_CERES_NUM_THREADS`, async vs `--repro-eval`)

**Rule:** fix all of the above *before* claiming “we beat OSS X”.

## Canonical evaluation flags (this repository)

Match `eval/regression_baselines.json` → `evo_ape_extra_args`:

```bash
evo_ape tum groundtruth.txt trajectory.txt \
  --align --correct_scale --t_max_diff 0.05
```

SimpleVisualSLAM writes `trajectory.txt` in **TUM** (`timestamp tx ty tz qx qy qz qw`) from the **build directory** when `run_mono` is launched with `cwd` there.

For a controlled window (example: 250 frames from the start):

```bash
./build/run_mono --tum "$TUM_RGBD_FREIBURG1_XYZ" \
  --max-frames 250 --repro-eval --no-viz [--depth] [--accel] \
  --reference-policy heuristic
# then evo_ape from build/: groundtruth is inside the dataset folder
```

Record: git commit, `./build/run_mono --version`, dataset path, exact CLI, and whether loop closing had a valid vocab.

## Suggested peers (same *class* of problem)

| Project | License | Notes |
|--------|---------|--------|
| **stella_vslam** | BSD-2-Clause | ORB + optional line/stereo; strong on EuRoC/TUM-style setups. Good *primary* numeric peer for “can we match classical feature SLAM on the same clip?”. |
| ORB-SLAM2 / ORB-SLAM3 | GPLv3 | Often stronger on many benchmarks, but **license and codebase size** differ; compare numbers if you want, but do not conflate with “replaceable module in a BSD product”. |
| OpenVINS, Kimera, etc. | Various | Useful baselines; check sensor assumptions (IMU-heavy, etc.). |

Pick **one peer and one scenario** first (e.g. TUM `freiburg1_xyz`, 250 frames, RGB-D + same `evo` flags). Expand only after the pipeline is stable.

## What “exceed” means here

Minimal defensible claim:

1. **Same** TUM sequence subdirectory and **same** frame window as SimpleVisualSLAM’s published gate (see `eval/regression_baselines.json` / `eval/leaderboard_suite.json`).
2. **Same** `evo_ape` alignment flags as above (or document intentional differences).
3. Peer system configured for **same modality** (e.g. both with depth if you compare RGB-D).
4. Report **mean ATE (m)** and, if possible, **repeat runs** (stdev)—this repo supports bitwise repeat under `--repro-eval` for internal regression; external tools may only get statistical repeatability.

Optional stricter bar: add **relative pose error (RPE)** with the same `evo` settings used in other papers—only after ATE is aligned.

## Recording results

Use a small table in your notes or paper supplement:

| System | Commit / version | Modality | Sequence + window | mean ATE (m) | evo flags |
|--------|------------------|----------|-------------------|--------------|-----------|
| SimpleVisualSLAM | `run_mono --version` + git SHA | mono / depth | e.g. xyz 0–250 | … | align + correct_scale + t_max_diff 0.05 |
| Peer | … | … | … | … | same |

## Step 1 — Baseline verification (this repository only)

Before involving stella_vslam or any other OSS, freeze **our** number on a preset that matches the regression harness:

1. Build `run_mono` (any directory; pass `BUILD=…` if not `build/`).
2. Install [evo](https://github.com/MichaelGrupp/evo) (`pip install evo`) so `evo_ape` is on `PATH`.
3. From the repo root:

```bash
bash scripts/verify_comparison_benchmark.sh xyz_depth
# or: xyz_mono | room_mono | room_depth
BUILD=build_test_tumcal bash scripts/verify_comparison_benchmark.sh xyz_mono
```

The script removes stale `trajectory*.txt` in `BUILD`, runs `--tum … --max-frames 250 --repro-eval --reference-policy heuristic` (adds `--depth` when the preset says so), then prints **mean ATE (m)** using `scripts/print_ate_mean.py`, which reads **`evo_ape_extra_args`** from `eval/regression_baselines.json`.

Copy that scalar plus `git rev-parse HEAD` and `./$BUILD/run_mono --version` into your comparison table; only then run the peer system with the **same** sequence folder, **same** modality, and **same** frame policy.

## Automation in this repo

- **Baseline one-shot:** `scripts/verify_comparison_benchmark.sh` + `scripts/print_ate_mean.py`.
- **Internal matrix:** `scripts/build_leaderboard.py` (methods × sequences).
- **Regression ceilings:** `scripts/check_regression_gate.py` (does not run external SLAM).

External runners stay in *their* repositories; this file defines how to line up numbers when you paste them next to ours.
