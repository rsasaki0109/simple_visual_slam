#!/bin/bash
# Comprehensive evaluation script for SimpleVisualSLAM
# Runs all modes on available TUM datasets and produces a comparison table.

set -uo pipefail

usage() {
    cat <<EOF
Usage: bash scripts/eval_all.sh [--repeat N]

Options:
  --repeat N   Run each dataset/mode pair N times and report aggregate mean/std.
  -h, --help   Show this help message.
EOF
}

REPEAT=1
while [ $# -gt 0 ]; do
    case "$1" in
        --repeat)
            if [ $# -lt 2 ]; then
                echo "Error: --repeat requires a positive integer argument"
                usage
                exit 1
            fi
            REPEAT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --repeat must be a positive integer"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
DATA_DIR="$SCRIPT_DIR/../data/tum"
RESULTS_DIR="$SCRIPT_DIR/../eval_results"

mkdir -p "$RESULTS_DIR"

RUN_MONO="$BUILD_DIR/run_mono"
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: build directory not found at $BUILD_DIR"
    echo "Build first: mkdir build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

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
echo "Repeats per mode: $REPEAT"
for d in "${DATASETS[@]}"; do
    echo "  - $(basename "$d")"
done
echo ""

# Results table
declare -A RESULTS
RUN_STATUS=""
RUN_MEAN=""
RUN_RMSE=""
RUN_MAX=""

aggregate_metrics() {
    local metrics_file="$1"
    awk '
        {
            mean_sum += $1
            mean_sq += $1 * $1
            rmse_sum += $2
            rmse_sq += $2 * $2
            max_sum += $3
            max_sq += $3 * $3
            n++
        }
        END {
            if (n == 0) {
                exit 1
            }
            mean_avg = mean_sum / n
            rmse_avg = rmse_sum / n
            max_avg = max_sum / n
            mean_var = mean_sq / n - mean_avg * mean_avg
            rmse_var = rmse_sq / n - rmse_avg * rmse_avg
            max_var = max_sq / n - max_avg * max_avg
            if (mean_var < 0) mean_var = 0
            if (rmse_var < 0) rmse_var = 0
            if (max_var < 0) max_var = 0
            printf "mean=%.6f std=%.6f rmse_mean=%.6f rmse_std=%.6f max_mean=%.6f max_std=%.6f runs=%d",
                mean_avg, sqrt(mean_var), rmse_avg, sqrt(rmse_var), max_avg, sqrt(max_var), n
        }
    ' "$metrics_file"
}

run_single_eval() {
    local dataset="$1"
    local mode="$2"
    local flags="$3"
    local run_idx="$4"
    local name="$(basename "$dataset")"
    local key="${name}__${mode}"
    local suffix=""
    local run_label="run ${run_idx}/${REPEAT}"

    if [ "$REPEAT" -gt 1 ]; then
        suffix="_run$(printf '%02d' "$run_idx")"
    fi

    local traj_file="$RESULTS_DIR/${key}${suffix}_trajectory.txt"
    local log_file="$RESULTS_DIR/${key}${suffix}_log.txt"

    RUN_STATUS=""
    RUN_MEAN=""
    RUN_RMSE=""
    RUN_MAX=""

    echo "--- Running: $name [$mode] $run_label ---"

    # Clean up previous trajectory files to avoid stale data
    rm -f trajectory.txt trajectory_online.txt trajectory_keyframes.txt map.bin

    local -a cmd=("$RUN_MONO" --tum "$dataset")
    if [ -n "$flags" ]; then
        read -r -a flag_parts <<< "$flags"
        cmd+=("${flag_parts[@]}")
    fi
    cmd+=(--no-viz)

    "${cmd[@]}" > "$log_file" 2>&1 &
    local run_pid=$!
    local start_time=$SECONDS
    local max_runtime_sec=600
    local trajectory_ready=0

    # Some runs finish processing but hang during shutdown. For evaluation, a flushed
    # trajectory is sufficient, so stop waiting once it has been fully written.
    while kill -0 "$run_pid" 2>/dev/null; do
        if [ -f trajectory.txt ] && [ -s trajectory.txt ] &&
           grep -q "Trajectory saved to trajectory.txt" "$log_file" 2>/dev/null; then
            trajectory_ready=1
            break
        fi

        if [ $((SECONDS - start_time)) -ge "$max_runtime_sec" ]; then
            break
        fi
        sleep 2
    done

    if kill -0 "$run_pid" 2>/dev/null; then
        if [ "$trajectory_ready" -eq 1 ]; then
            kill -INT "$run_pid" 2>/dev/null || true
            sleep 2
        fi
        if kill -0 "$run_pid" 2>/dev/null; then
            kill -TERM "$run_pid" 2>/dev/null || true
            sleep 2
        fi
        if kill -0 "$run_pid" 2>/dev/null; then
            kill -KILL "$run_pid" 2>/dev/null || true
        fi
    fi
    wait "$run_pid" 2>/dev/null || true

    # Check if trajectory was produced
    if [ -f trajectory.txt ] && [ -s trajectory.txt ]; then
        cp trajectory.txt "$traj_file"
    else
        RUN_STATUS="FAILED (no trajectory output)"
        return
    fi

    # Evaluate with evo
    local gt="$dataset/groundtruth.txt"
    if [ ! -f "$gt" ]; then
        RUN_STATUS="NO_GT"
        return
    fi

    local ate_output
    ate_output=$(evo_ape tum "$gt" "$traj_file" --align --correct_scale --t_max_diff 0.05 2>&1)

    # Check for errors (no matching timestamps, etc.)
    if echo "$ate_output" | grep -q "\[ERROR\]"; then
        echo "  evo_ape error: $(echo "$ate_output" | grep "\[ERROR\]")"
        RUN_STATUS="EVO_ERROR"
        return
    fi

    # Parse metrics (tab-separated: label\tvalue)
    local rmse=$(echo "$ate_output" | grep -E "^\s+rmse" | awk '{print $2}')
    local mean=$(echo "$ate_output" | grep -E "^\s+mean" | awk '{print $2}')
    local max_val=$(echo "$ate_output" | grep -E "^\s+max" | awk '{print $2}')

    if [ -z "$rmse" ] || [ -z "$mean" ] || [ -z "$max_val" ]; then
        RUN_STATUS="PARSE_ERROR"
        return
    fi

    RUN_STATUS="OK"
    RUN_MEAN="$mean"
    RUN_RMSE="$rmse"
    RUN_MAX="$max_val"
    echo "  ATE: mean=$mean rmse=$rmse max=$max_val"
}

run_eval() {
    local dataset="$1"
    local mode="$2"
    local flags="$3"
    local name="$(basename "$dataset")"
    local key="${name}__${mode}"
    local metrics_file="$RESULTS_DIR/${key}_metrics.tsv"
    local success_count=0

    : > "$metrics_file"

    for ((run_idx = 1; run_idx <= REPEAT; ++run_idx)); do
        run_single_eval "$dataset" "$mode" "$flags" "$run_idx"
        if [ "$RUN_STATUS" = "OK" ]; then
            printf "%s\t%s\t%s\n" "$RUN_MEAN" "$RUN_RMSE" "$RUN_MAX" >> "$metrics_file"
            success_count=$((success_count + 1))
        else
            echo "  $RUN_STATUS"
        fi
    done

    if [ "$success_count" -eq 0 ]; then
        RESULTS[$key]="FAILED (0/${REPEAT} successful)"
        return
    fi

    if [ "$REPEAT" -eq 1 ]; then
        RESULTS[$key]="mean=${RUN_MEAN} rmse=${RUN_RMSE} max=${RUN_MAX}"
        return
    fi

    local aggregate
    aggregate="$(aggregate_metrics "$metrics_file")"
    RESULTS[$key]="${aggregate} success=${success_count}/${REPEAT}"
    echo "  Aggregate: ${RESULTS[$key]}"
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
printf "%-35s %-12s %s\n" "Dataset" "Mode" "ATE summary"
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
    echo "Repeats per mode: $REPEAT"
    echo ""
    printf "%-35s %-12s %s\n" "Dataset" "Mode" "ATE summary"
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
if [ "$REPEAT" -gt 1 ]; then
    echo "Per-run metrics files: $RESULTS_DIR/*_metrics.tsv"
fi
