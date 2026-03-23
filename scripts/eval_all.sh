#!/bin/bash
# Comprehensive evaluation script for SimpleVisualSLAM
# Runs all modes on available TUM datasets and produces a comparison table.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$(cd "$SCRIPT_DIR/../build" && pwd)"
DATA_DIR="$(cd "$SCRIPT_DIR/../data/tum" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../eval_results"

mkdir -p "$RESULTS_DIR"

RUN_MONO="$BUILD_DIR/run_mono"
if [ ! -f "$RUN_MONO" ]; then
    echo "Error: run_mono not found at $RUN_MONO"
    echo "Build first: cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# Must run from build dir for ORBvoc.txt and other data paths
cd "$BUILD_DIR"

# Available datasets
DATASETS=()
for d in "$DATA_DIR"/rgbd_dataset_freiburg*; do
    [ -d "$d" ] && DATASETS+=("$d")
done

if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "No TUM datasets found in $DATA_DIR"
    exit 1
fi

echo "============================================"
echo " SimpleVisualSLAM Evaluation"
echo "============================================"
echo ""
echo "Datasets: ${#DATASETS[@]}"
for d in "${DATASETS[@]}"; do
    echo "  - $(basename "$d")"
done
echo ""

# Results table
declare -A RESULTS

run_eval() {
    local dataset="$1"
    local mode="$2"
    local flags="$3"
    local name="$(basename "$dataset")"
    local key="${name}__${mode}"
    local traj_file="$RESULTS_DIR/${key}_trajectory.txt"

    echo "--- Running: $name [$mode] ---"

    # Clean up previous trajectory files to avoid stale data
    rm -f trajectory.txt trajectory_online.txt trajectory_keyframes.txt map.bin

    # Run SLAM (allow non-zero exit codes including segfaults)
    timeout 600 "$RUN_MONO" --tum "$dataset" $flags --no-viz > "$RESULTS_DIR/${key}_log.txt" 2>&1 || true

    # Check if trajectory was produced
    if [ -f trajectory.txt ] && [ -s trajectory.txt ]; then
        cp trajectory.txt "$traj_file"
    else
        echo "  FAILED (no trajectory output)"
        RESULTS[$key]="FAILED"
        return
    fi

    # Evaluate with evo
    local gt="$dataset/groundtruth.txt"
    if [ ! -f "$gt" ]; then
        RESULTS[$key]="NO_GT"
        return
    fi

    local ate_output
    ate_output=$(evo_ape tum "$gt" "$traj_file" --align --correct_scale --t_max_diff 0.05 2>&1)

    # Check for errors (no matching timestamps, etc.)
    if echo "$ate_output" | grep -q "\[ERROR\]"; then
        echo "  evo_ape error: $(echo "$ate_output" | grep "\[ERROR\]")"
        RESULTS[$key]="EVO_ERROR"
        return
    fi

    # Parse metrics (tab-separated: label\tvalue)
    local rmse=$(echo "$ate_output" | grep -E "^\s+rmse" | awk '{print $2}')
    local mean=$(echo "$ate_output" | grep -E "^\s+mean" | awk '{print $2}')
    local max_val=$(echo "$ate_output" | grep -E "^\s+max" | awk '{print $2}')

    RESULTS[$key]="mean=${mean} rmse=${rmse} max=${max_val}"
    echo "  ATE: mean=$mean rmse=$rmse max=$max_val"
}

# Run evaluations for each dataset
for dataset in "${DATASETS[@]}"; do
    name="$(basename "$dataset")"

    # Mode 1: Pure monocular (no depth, no accel)
    run_eval "$dataset" "mono" ""

    # Mode 2: Depth only
    if [ -f "$dataset/depth.txt" ]; then
        run_eval "$dataset" "depth" "--depth"
    fi

    # Mode 3: Depth + Accelerometer
    if [ -f "$dataset/depth.txt" ] && [ -f "$dataset/accelerometer.txt" ]; then
        run_eval "$dataset" "depth_accel" "--depth --accel"
    fi

    echo ""
done

# Print summary table
echo ""
echo "============================================"
echo " RESULTS SUMMARY"
echo "============================================"
printf "%-35s %-12s %s\n" "Dataset" "Mode" "ATE (mean/rmse/max)"
printf "%-35s %-12s %s\n" "-----------------------------------" "------------" "-------------------"

for dataset in "${DATASETS[@]}"; do
    name="$(basename "$dataset")"
    for mode in mono depth depth_accel; do
        key="${name}__${mode}"
        if [ -n "${RESULTS[$key]}" ]; then
            printf "%-35s %-12s %s\n" "$name" "$mode" "${RESULTS[$key]}"
        fi
    done
done

# Save summary
SUMMARY_FILE="$RESULTS_DIR/summary.txt"
{
    echo "SimpleVisualSLAM Evaluation Summary"
    echo "Date: $(date -Iseconds)"
    echo ""
    printf "%-35s %-12s %s\n" "Dataset" "Mode" "ATE (mean/rmse/max)"
    printf "%-35s %-12s %s\n" "---" "---" "---"
    for dataset in "${DATASETS[@]}"; do
        name="$(basename "$dataset")"
        for mode in mono depth depth_accel; do
            key="${name}__${mode}"
            if [ -n "${RESULTS[$key]}" ]; then
                printf "%-35s %-12s %s\n" "$name" "$mode" "${RESULTS[$key]}"
            fi
        done
    done
} > "$SUMMARY_FILE"

echo ""
echo "Results saved to: $SUMMARY_FILE"
echo "Trajectory files in: $RESULTS_DIR/"
