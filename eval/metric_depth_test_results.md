# Metric Depth Pipeline Test Results

Date: 2026-04-14

## Summary

- Build succeeded with `cmake -S . -B build_codex3 -G Ninja -DBUILD_TESTS=ON -DUSE_DEPTH_DL=ON && cmake --build build_codex3 -j$(nproc)`.
- The metric ONNX path works with `models/depth_anything_v2_metric_indoor_small.onnx`.
- `rgbd_dataset_freiburg1_xyz`, `50` frames: exit `0`, trajectory `50` poses, ATE `0.01093245 m`.
- `rgbd_dataset_freiburg1_xyz`, `250` frames: exit `0`, trajectory `250` poses, ATE `0.06528716 m`.
- `rgbd_dataset_freiburg1_room`, `50` frames: exit `0`, trajectory `50` poses, ATE `0.01712747 m`.
- The two suggested Hugging Face indoor ONNX URLs both failed with `401 Unauthorized`.
- A public alternative metric ONNX download succeeded for `sollaholla/depth-anything-v2-metric-vits-vkitti`, but it is an outdoor model and was not used for the indoor TUM evaluations.

## Model Search And Download Attempts

### Tried And Failed

1. Official indoor ONNX URL

```bash
wget -O models/depth_anything_v2_metric_indoor.onnx \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small/resolve/main/onnx/model.onnx"
```

Result:

- Failed with `401 Unauthorized`

2. `onnx-community` indoor ONNX URL

```bash
wget -O models/depth_anything_v2_metric_indoor.onnx \
  "https://huggingface.co/onnx-community/depth-anything-v2-metric-indoor-small/resolve/main/onnx/model.onnx"
```

Result:

- Failed with `401 Unauthorized`

3. Hugging Face API searches

```bash
curl -s 'https://huggingface.co/api/models?search=depth-anything-v2-metric-indoor&limit=20'
curl -s 'https://huggingface.co/api/models?search=metric%20depth%20onnx&limit=20'
curl -s 'https://huggingface.co/api/models?search=depth-anything-v2%20onnx-community&limit=50'
curl -s 'https://huggingface.co/api/models?search=metric%20depth%20depth-anything&limit=100'
```

Result:

- Found official transformer checkpoints such as `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`
- Did not find a public official indoor ONNX repo at the expected `depth-anything/.../onnx/model.onnx` or `onnx-community/.../onnx/model.onnx` paths
- Found a public ONNX-tagged alternative metric model: `sollaholla/depth-anything-v2-metric-vits-vkitti`

### Worked

1. Public alternative metric ONNX model

```bash
wget -O /tmp/depth_anything_v2_metric_vits_vkitti_518x518.onnx \
  "https://huggingface.co/sollaholla/depth-anything-v2-metric-vits-vkitti/resolve/main/depth_anything_v2_metric_vits_vkitti_518x518.onnx"
wget -O models/depth_anything_v2_metric_vits_vkitti_518x518.onnx.data \
  "https://huggingface.co/sollaholla/depth-anything-v2-metric-vits-vkitti/resolve/main/depth_anything_v2_metric_vits_vkitti_518x518.onnx.data"
cp /tmp/depth_anything_v2_metric_vits_vkitti_518x518.onnx \
  models/depth_anything_v2_metric_vits_vkitti_518x518.onnx
```

Result:

- Succeeded
- Saved files:
  - `models/depth_anything_v2_metric_vits_vkitti_518x518.onnx`
  - `models/depth_anything_v2_metric_vits_vkitti_518x518.onnx.data`

2. Existing local indoor metric ONNX candidate found in the repo

Result:

- `models/depth_anything_v2_metric_indoor_small.onnx` already existed locally
- Size: `95M`
- SHA256: `afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c`
- This was used for evaluation because it is the indoor metric ONNX already present in the workspace and it loaded successfully in the metric-depth pipeline

## Build

Command:

```bash
cmake -S . -B build_codex3 -G Ninja -DBUILD_TESTS=ON -DUSE_DEPTH_DL=ON && \
cmake --build build_codex3 -j$(nproc)
```

Result:

- Exit status: `0`
- Built targets included `run_mono` and `svslam_tests`

## Evaluation Runs

### xyz, 50 Frames

Commands:

```bash
rm -f trajectory.txt
build_codex3/run_mono --tum data/tum/rgbd_dataset_freiburg1_xyz \
  --metric-depth-model models/depth_anything_v2_metric_indoor_small.onnx \
  --max-frames 50 --no-viz
python3 scripts/print_ate_mean.py \
  data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt trajectory.txt
```

Result:

- Exit status: `0`
- Metric model load: `SUCCESS`
- Depth initialization: `SUCCESS`
- Processed frames: `50`
- Trajectory produced: `YES`
- Trajectory file lines: `51`
- ATE: `0.01093245 m`

Relevant tail:

```text
Finished processing.
Processed frames: 50 (skipped 0)
Trajectory saved to trajectory.txt (50 poses)
Trajectory saved to trajectory_online.txt
LocalMapping thread stopped.
LoopClosing thread stopped.
Saving map to map.bin...
MapIO: Saved 11 KFs and 5048 LMs to map.bin
Map saved successfully.
Keyframe trajectory saved to trajectory_keyframes.txt (11 keyframes)
```

### xyz, 250 Frames

Commands:

```bash
rm -f trajectory.txt
build_codex3/run_mono --tum data/tum/rgbd_dataset_freiburg1_xyz \
  --metric-depth-model models/depth_anything_v2_metric_indoor_small.onnx \
  --max-frames 250 --no-viz
python3 scripts/print_ate_mean.py \
  data/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt trajectory.txt
```

Result:

- Exit status: `0`
- Metric model load: `SUCCESS`
- Processed frames: `250`
- Trajectory produced: `YES`
- Trajectory file lines: `251`
- ATE: `0.06528716 m`

Relevant tail:

```text
Finished processing.
Processed frames: 250 (skipped 0)
Trajectory saved to trajectory.txt (250 poses)
Trajectory saved to trajectory_online.txt
LocalMapping thread stopped.
LoopClosing thread stopped.
Saving map to map.bin...
MapIO: Saved 54 KFs and 16119 LMs to map.bin
Map saved successfully.
Keyframe trajectory saved to trajectory_keyframes.txt (54 keyframes)
```

### room, 50 Frames

Commands:

```bash
rm -f trajectory.txt
build_codex3/run_mono --tum data/tum/rgbd_dataset_freiburg1_room \
  --metric-depth-model models/depth_anything_v2_metric_indoor_small.onnx \
  --max-frames 50 --no-viz
python3 scripts/print_ate_mean.py \
  data/tum/rgbd_dataset_freiburg1_room/groundtruth.txt trajectory.txt
```

Result:

- Exit status: `0`
- Metric model load: `SUCCESS`
- Processed frames: `50`
- Trajectory produced: `YES`
- Trajectory file lines: `51`
- ATE: `0.01712747 m`

Relevant tail:

```text
Finished processing.
Processed frames: 50 (skipped 0)
Trajectory saved to trajectory.txt (50 poses)
Trajectory saved to trajectory_online.txt
LocalMapping thread stopped.
LoopClosing thread stopped.
Saving map to map.bin...
MapIO: Saved 16 KFs and 6217 LMs to map.bin
Map saved successfully.
Keyframe trajectory saved to trajectory_keyframes.txt (16 keyframes)
```

## Notes

- The requested filename `models/depth_anything_v2_metric_indoor.onnx` could not be populated from the two suggested Hugging Face URLs because both returned `401 Unauthorized` on `2026-04-14`.
- The evaluated metric model path was `models/depth_anything_v2_metric_indoor_small.onnx`.
- The metric-depth pipeline is now verified end-to-end on indoor TUM data with a true metric ONNX model, unlike the earlier failed attempt that used the relative-depth `depth_anything_v2_small.onnx` under `--metric-depth-model`.
