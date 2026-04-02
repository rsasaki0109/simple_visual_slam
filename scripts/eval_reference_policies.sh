#!/bin/bash

set -euo pipefail

usage() {
    cat <<EOF
Usage: bash scripts/eval_reference_policies.sh [--repeat N] [--corpus path]

Options:
  --repeat N    Run each case/policy pair N times. Default: 1
  --corpus PATH Path to the real-trace corpus TSV.
  --mode MODE   Only run rows whose mode matches MODE.
  --policy NAME Only run a single policy.
  --output PATH Write results CSV to PATH.
  --no-repro    Disable reproducible evaluation mode.
  -h, --help    Show this help message.
EOF
}

REPEAT=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CORPUS_FILE="$ROOT_DIR/experiments/reference_keyframe/real_trace_corpus.tsv"
BUILD_DIR="$ROOT_DIR/build"
DATA_DIR="$ROOT_DIR/data/tum"
OUTPUT_DIR="$ROOT_DIR/eval_results/reference_keyframe_policy"
RUN_MONO="$BUILD_DIR/run_mono"
POLICIES=(heuristic score pipeline)
MODE_FILTER=""
POLICY_FILTER=""
RESULTS_CSV=""
REPRO_EVAL=1

while [ $# -gt 0 ]; do
    case "$1" in
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        --corpus)
            CORPUS_FILE="$2"
            shift 2
            ;;
        --mode)
            MODE_FILTER="$2"
            shift 2
            ;;
        --policy)
            POLICY_FILTER="$2"
            shift 2
            ;;
        --output)
            RESULTS_CSV="$2"
            shift 2
            ;;
        --no-repro)
            REPRO_EVAL=0
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$REPEAT" =~ ^[1-9][0-9]*$ ]]; then
    echo "--repeat must be a positive integer"
    exit 1
fi

if [ ! -f "$RUN_MONO" ]; then
    echo "Missing binary: $RUN_MONO"
    echo "Build first with: cmake -S . -B build && cmake --build build -j4 --target run_mono"
    exit 1
fi

if [[ "$CORPUS_FILE" != /* ]]; then
    CORPUS_FILE="$ROOT_DIR/$CORPUS_FILE"
fi

if [ ! -f "$CORPUS_FILE" ]; then
    echo "Missing corpus file: $CORPUS_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [ -z "$RESULTS_CSV" ]; then
    RESULTS_CSV="$OUTPUT_DIR/real_trace_metrics.csv"
elif [[ "$RESULTS_CSV" != /* ]]; then
    RESULTS_CSV="$ROOT_DIR/$RESULTS_CSV"
fi
mkdir -p "$(dirname "$RESULTS_CSV")"
echo "policy,case_id,dataset,mode,skip_frames,max_frames,repro_eval,run_idx,status,mean,rmse,max,trajectory_file,log_file" > "$RESULTS_CSV"

cd "$BUILD_DIR"

run_case() {
    local policy="$1"
    local case_id="$2"
    local dataset_name="$3"
    local mode="$4"
    local skip_frames="$5"
    local max_frames="$6"
    local run_idx="$7"

    local dataset_path="$DATA_DIR/$dataset_name"
    local flags=()
    if [ "$mode" = "depth" ]; then
        flags+=(--depth)
    elif [ "$mode" = "depth_accel" ]; then
        flags+=(--depth --accel)
    fi

    local stem="${policy}__${case_id}__run$(printf '%02d' "$run_idx")"
    local log_file="$OUTPUT_DIR/${stem}.log"
    local traj_file="$OUTPUT_DIR/${stem}_trajectory.txt"

    rm -f trajectory.txt trajectory_online.txt trajectory_keyframes.txt map.bin

    local -a cmd=(
        "$RUN_MONO"
        --tum "$dataset_path"
        --reference-policy "$policy"
        --skip-frames "$skip_frames"
        --max-frames "$max_frames"
        --no-viz
    )
    if [ "$REPRO_EVAL" -eq 1 ]; then
        cmd+=(--repro-eval)
    fi
    cmd+=("${flags[@]}")

    echo "--- $policy :: $case_id :: run $run_idx/$REPEAT ---"
    "${cmd[@]}" > "$log_file" 2>&1

    if [ ! -f trajectory.txt ] || [ ! -s trajectory.txt ]; then
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$policy" "$case_id" "$dataset_name" "$mode" "$skip_frames" "$max_frames" "$REPRO_EVAL" "$run_idx" \
            "FAILED_NO_TRAJECTORY" "" "" "" "" "$log_file" >> "$RESULTS_CSV"
        return
    fi

    cp trajectory.txt "$traj_file"

    local gt="$dataset_path/groundtruth.txt"
    local ate_output
    ate_output=$(evo_ape tum "$gt" "$traj_file" --align --correct_scale --t_max_diff 0.05 2>&1 || true)
    if echo "$ate_output" | grep -q "\[ERROR\]"; then
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$policy" "$case_id" "$dataset_name" "$mode" "$skip_frames" "$max_frames" "$REPRO_EVAL" "$run_idx" \
            "EVO_ERROR" "" "" "" "$traj_file" "$log_file" >> "$RESULTS_CSV"
        return
    fi

    local rmse
    local mean
    local max_val
    rmse=$(echo "$ate_output" | grep -E "^\s+rmse" | awk '{print $2}')
    mean=$(echo "$ate_output" | grep -E "^\s+mean" | awk '{print $2}')
    max_val=$(echo "$ate_output" | grep -E "^\s+max" | awk '{print $2}')

    if [ -z "$rmse" ] || [ -z "$mean" ] || [ -z "$max_val" ]; then
        printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "$policy" "$case_id" "$dataset_name" "$mode" "$skip_frames" "$max_frames" "$REPRO_EVAL" "$run_idx" \
            "PARSE_ERROR" "" "" "" "$traj_file" "$log_file" >> "$RESULTS_CSV"
        return
    fi

    echo "  ATE: mean=$mean rmse=$rmse max=$max_val"
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$policy" "$case_id" "$dataset_name" "$mode" "$skip_frames" "$max_frames" "$REPRO_EVAL" "$run_idx" \
        "OK" "$mean" "$rmse" "$max_val" "$traj_file" "$log_file" >> "$RESULTS_CSV"
}

while IFS=$'\t' read -r case_id dataset_name mode skip_frames max_frames; do
    if [ "$case_id" = "case_id" ]; then
        continue
    fi
    if [ -n "$MODE_FILTER" ] && [ "$mode" != "$MODE_FILTER" ]; then
        continue
    fi
    for policy in "${POLICIES[@]}"; do
        if [ -n "$POLICY_FILTER" ] && [ "$policy" != "$POLICY_FILTER" ]; then
            continue
        fi
        for ((run_idx = 1; run_idx <= REPEAT; ++run_idx)); do
            run_case "$policy" "$case_id" "$dataset_name" "$mode" "$skip_frames" "$max_frames" "$run_idx"
        done
    done
done < "$CORPUS_FILE"

echo ""
echo "Real-trace policy metrics saved to: $RESULTS_CSV"
