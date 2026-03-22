#!/bin/bash
# ATE evaluation script for SimpleVisualSLAM
# Usage: ./scripts/eval_ate.sh

set -e

GT_XYZ="../data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt"
GT_ROOM="../data/tum/rgbd_dataset_freiburg1_room/groundtruth.txt"

echo "============================================="
echo " SimpleVisualSLAM ATE Evaluation"
echo "============================================="

for dataset in xyz room; do
    if [ "$dataset" = "xyz" ]; then
        GT=$GT_XYZ
    else
        GT=$GT_ROOM
    fi

    for mode in mono depth; do
        TRAJ="../eval_${mode}_${dataset}.txt"
        if [ ! -f "$TRAJ" ]; then
            echo "[$dataset / $mode] SKIP - $TRAJ not found"
            continue
        fi

        echo ""
        echo "--- $dataset / $mode ---"
        evo_ape tum "$GT" "$TRAJ" --align --correct_scale -r trans_part 2>&1 | grep -E "(rmse|mean|median|std|min|max|Aligning)" || echo "  evo_ape failed"
    done

    # Also try with Sim3 alignment for mono (scale is unknown)
    TRAJ_MONO="../eval_mono_${dataset}.txt"
    if [ -f "$TRAJ_MONO" ]; then
        echo ""
        echo "--- $dataset / mono (Sim3 align) ---"
        evo_ape tum "$GT" "$TRAJ_MONO" --align --correct_scale -r trans_part 2>&1 | grep -E "(rmse|mean|median|std|min|max)" || echo "  evo_ape failed"
    fi

    # Depth: should be metric, try without scale correction too
    TRAJ_DEPTH="../eval_depth_${dataset}.txt"
    if [ -f "$TRAJ_DEPTH" ]; then
        echo ""
        echo "--- $dataset / depth (SE3 align only, no scale correction) ---"
        evo_ape tum "$GT" "$TRAJ_DEPTH" --align -r trans_part 2>&1 | grep -E "(rmse|mean|median|std|min|max)" || echo "  evo_ape failed"
    fi
done

echo ""
echo "============================================="
echo " Done"
echo "============================================="
