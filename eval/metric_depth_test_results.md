# Metric Depth Pipeline Test Results

Date: 2026-04-14

## Summary

- Download succeeded: `models/depth_anything_v2_small.onnx` fetched successfully (`99,060,839` bytes).
- Build succeeded: `cmake -S . -B build_codex3 -G Ninja -DBUILD_TESTS=ON -DUSE_DEPTH_DL=ON`
- Relative depth pipeline works: `YES`
- Metric depth pipeline works: `NO`

## Relative Depth Test

Command:

```bash
rm -f trajectory.txt
build_codex3/run_mono --tum data/tum/rgbd_dataset_freiburg1_xyz --depth-model models/depth_anything_v2_small.onnx --max-frames 50 --no-viz
```

Result:

- Exit status: `0`
- Processed frames: `50`
- Trajectory produced: `YES` (`trajectory.txt`, 50 poses; file had 51 lines)
- ATE: `0.00464917 m`

Relevant tail output:

```text
Finished processing.
Processed frames: 50 (skipped 0)
Trajectory saved to trajectory.txt (50 poses)
Trajectory saved to trajectory_online.txt
LocalMapping: BA on 9 KFs and 800 LMs.
BA: Added 1973 depth prior residuals
Ceres Solver Report: Iterations: 16, Initial cost: 8.075023e+03, Final cost: 3.888414e+03, Termination: NO_CONVERGENCE
LocalMapping thread stopped.
LoopClosing thread stopped.
Saving map to map.bin...
MapIO: Saved 9 KFs and 5041 LMs to map.bin
Map saved successfully.
Keyframe trajectory saved to trajectory_keyframes.txt (9 keyframes)
```

ATE was computed with:

```bash
python3 scripts/print_ate_mean.py data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt trajectory.txt
```

## Metric Depth Test

Command:

```bash
rm -f trajectory.txt
build_codex3/run_mono --tum data/tum/rgbd_dataset_freiburg1_xyz --metric-depth-model models/depth_anything_v2_small.onnx --max-frames 50 --no-viz
```

Result:

- Exit status: `134`
- Trajectory produced: `NO`
- ATE: `N/A`

Errors encountered:

```text
Loading metric DL depth model: models/depth_anything_v2_small.onnx
terminate called after throwing an instance of 'std::length_error'
  what():  cannot create std::vector larger than max_size()
```

## Notes

- The downloaded model is the Depth Anything V2 small ONNX model from Hugging Face.
- It works for the relative-depth verification path (`--depth-model`).
- It does not work for the metric-depth path (`--metric-depth-model`) in the current pipeline; the process aborts during model loading before any trajectory is produced.
