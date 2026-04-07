#!/usr/bin/env bash
# Comparison verification (step 1): run SimpleVisualSLAM on a fixed TUM window and print mean ATE
# using the same evo_ape flags as eval/regression_baselines.json.
#
# Usage (from repo root):
#   bash scripts/verify_comparison_benchmark.sh xyz_depth
#   bash scripts/verify_comparison_benchmark.sh room_mono
#   BUILD=build_test_tumcal bash scripts/verify_comparison_benchmark.sh xyz_mono
#
# Presets align with regression gate names (250-frame head, --repro-eval, heuristic policy).
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD:-$ROOT/build}"
DATA_TUM="${DATA_TUM:-$ROOT/data/tum}"
PRESET="${1:-}"

usage() {
  sed -n '2,20p' "$0" | tail -n +1
  echo ""
  echo "Presets: xyz_mono | xyz_depth | room_mono | room_depth"
  echo "Requires: evo_ape on PATH, dataset under data/tum/, executable run_mono in BUILD."
  exit 2
}

[[ -n "$PRESET" ]] || usage

command -v evo_ape >/dev/null || { echo "evo_ape not found (pip install evo)"; exit 1; }
[[ -x "$BUILD/run_mono" ]] || { echo "Missing $BUILD/run_mono — build first (cmake --build ... run_mono)"; exit 1; }
[[ -d "$DATA_TUM" ]] || { echo "Missing $DATA_TUM"; exit 1; }

case "$PRESET" in
  xyz_mono)
    SEQ_DIR="$DATA_TUM/rgbd_dataset_freiburg1_xyz"
    FLAGS=(--max-frames 250 --repro-eval --no-viz --reference-policy heuristic)
    ;;
  xyz_depth)
    SEQ_DIR="$DATA_TUM/rgbd_dataset_freiburg1_xyz"
    FLAGS=(--depth --max-frames 250 --repro-eval --no-viz --reference-policy heuristic)
    ;;
  room_mono)
    SEQ_DIR="$DATA_TUM/rgbd_dataset_freiburg1_room"
    FLAGS=(--max-frames 250 --repro-eval --no-viz --reference-policy heuristic)
    ;;
  room_depth)
    SEQ_DIR="$DATA_TUM/rgbd_dataset_freiburg1_room"
    FLAGS=(--depth --max-frames 250 --repro-eval --no-viz --reference-policy heuristic)
    ;;
  *)
    usage
    ;;
esac

GT="$SEQ_DIR/groundtruth.txt"
[[ -d "$SEQ_DIR" ]] || { echo "Missing sequence: $SEQ_DIR"; exit 1; }
[[ -f "$GT" ]] || { echo "Missing $GT"; exit 1; }

echo "=== comparison verification: SimpleVisualSLAM ==="
echo "preset:      $PRESET"
echo "build:       $BUILD"
echo "sequence:    $SEQ_DIR"
if command -v git >/dev/null && git -C "$ROOT" rev-parse --short HEAD >/dev/null 2>&1; then
  echo "git HEAD:    $(git -C "$ROOT" rev-parse --short HEAD)"
fi
echo "run_mono:    $("$BUILD/run_mono" --version 2>&1 | tr -d '\r')"
echo ""
echo "Running SLAM (cwd=$BUILD) …"
(
  cd "$BUILD"
  rm -f trajectory.txt trajectory_online.txt trajectory_keyframes.txt map.bin
  ./run_mono --tum "$SEQ_DIR" "${FLAGS[@]}"
)

TRAJ="$BUILD/trajectory.txt"
[[ -s "$TRAJ" ]] || { echo "No trajectory written to $TRAJ"; exit 1; }

echo ""
echo "Mean ATE (m), evo_ape flags = regression_baselines evo_ape_extra_args:"
MEAN="$(python3 "$ROOT/scripts/print_ate_mean.py" "$GT" "$TRAJ")"
echo "$MEAN"
echo ""
echo "Peer OSS: run the same sequence window (0–250 frames policy), same modality,"
echo "then evo_ape with: --align --correct_scale --t_max_diff 0.05"
echo "See eval/comparison_protocol.md"
