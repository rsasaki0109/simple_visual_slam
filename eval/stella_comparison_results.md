# stella_vslam Comparison

## Fair Head-250 Comparison

| Scenario | Modality | SimpleVisualSLAM ATE (m) | stella_vslam ATE (m, head-250) | Delta (Simple - stella) (m) | Winner |
| --- | --- | ---: | ---: | ---: | --- |
| `xyz_depth` | RGB-D | 0.01136912 | 0.00889256 | 0.00247656 | `stella_vslam` |
| `xyz_mono` | Mono | 0.04827879 | 0.01413570 | 0.03414309 | `stella_vslam` |
| `room_depth` | RGB-D | 0.08606639 | 0.02110508 | 0.06496131 | `stella_vslam` |
| `room_mono` | Mono | 0.19982044 | 0.02743546 | 0.17238498 | `stella_vslam` |

## Protocol Details

- Generated: `2026-04-13T13:11:19+09:00`
- SimpleVisualSLAM commit: `6429d9b9ba34697c668b474acac8ca456653f8fe`
- SimpleVisualSLAM version: `SimpleVisualSLAM 0.1.0`
- Build: `build_codex`
- SimpleVisualSLAM run window: first `250` tracked frames from the start of each sequence
- SimpleVisualSLAM common flags: `--max-frames 250 --repro-eval --no-viz --reference-policy heuristic`
- Depth scenarios add: `--depth`
- `evo_ape` flags: `--align --correct_scale --t_max_diff 0.05`
- SimpleVisualSLAM loop closing: disabled in all four runs. The logs report `Repro eval mode: ENABLED (synchronous local mapping, loop closing disabled)` and `LoopClosing: vocab file not found: ORBvoc.txt (loop closing disabled)` when launched from `build_codex/`.
- Provided `stella_eval/*/frame_trajectory.txt` artifacts were not canonical 250-frame windows: `xyz_depth=798`, `xyz_mono=792`, `room_depth=1362`, `room_mono=548` poses.
- The comparison table above therefore uses the first `250` poses from each provided stella trajectory as the closest fair head-window available from the supplied artifacts.
- Raw full-trajectory stella ATEs from the provided files were: `xyz_depth=0.00832365`, `xyz_mono=0.01076242`, `room_depth=0.03321209`, `room_mono=0.02600812`.
- stella_vslam loop-closing / vocab status could not be verified from the supplied `stella_eval` artifacts alone.

## Loss Hypotheses

- `xyz_depth`: Slightly higher local RGB-D pose noise from the current heuristic keyframe/depth stack than stella_vslam's more mature ORB + local BA pipeline on this short translation-heavy clip.
- `xyz_mono`: Monocular initialization and early map maintenance drift more than stella_vslam's ORB front-end/local optimization, producing a much larger head-250 pose error.
- `room_depth`: With loop closing disabled in `--repro-eval`, the room sequence accumulates much more drift through revisits than stella_vslam on the same early trajectory segment.
- `room_mono`: Monocular scale drift plus no loop closure is especially costly on `room_mono`, where correspondence support thins out and the map degrades much earlier than stella_vslam.

## Loop-Enabled Comparison

| Scenario | Modality | SimpleVisualSLAM Median ATE (m, 3 runs) | Raw ATEs (m) | stella_vslam ATE (m, head-250) | Delta (Simple - stella) (m) | Winner |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `xyz_depth` | RGB-D | 0.01139951 | `0.01139951, 0.01108985, 0.01142678` | 0.00889256 | 0.00250695 | `stella_vslam` |
| `xyz_mono` | Mono | 0.04529873 | `0.03893269, 0.04756870, 0.04529873` | 0.01413570 | 0.03116303 | `stella_vslam` |
| `room_depth` | RGB-D | 0.08255800 | `0.08461199, 0.08255800, 0.08083409` | 0.02110508 | 0.06145292 | `stella_vslam` |
| `room_mono` | Mono | 0.20024479 | `0.19253079, 0.20024479, 0.28880425` | 0.02743546 | 0.17280933 | `stella_vslam` |

- Generated: `2026-04-13T14:06:46+09:00`
- SimpleVisualSLAM commit: `244bb56288804ca86f9758f9e845c7d200d66796`
- SimpleVisualSLAM version: `SimpleVisualSLAM 0.1.0`
- SimpleVisualSLAM common flags: `--max-frames 250 --no-viz data/ORBvoc.txt`; depth scenarios add `--depth`
- Loop closing enabled: yes. These runs omitted `--repro-eval`, loaded `data/ORBvoc.txt`, and started the `LoopClosing` thread.
- Loop detections observed inside the 250-frame window: `xyz_depth=2/2/2`, `xyz_mono=1/1/1`, `room_depth=0/2/2`, `room_mono=0/0/0`.
- Gap change vs the earlier `--repro-eval` comparison: `xyz_depth` widened by `0.00003039 m`, `xyz_mono` narrowed by `0.00298006 m`, `room_depth` narrowed by `0.00350839 m`, `room_mono` widened by `0.00042435 m`.
- Conclusion: enabling loop closing within the first `250` tracked frames did not materially close the gap to `stella_vslam`; the biggest improvement was `room_depth`, but SimpleVisualSLAM still trails by `0.06145292 m`.

## 600-Frame Validation

| Run | ATE (m) | Loop Detections | Pending Loop Corrections Applied | Stabilization Rejections | Forced Reference Refreshes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rep1` | 0.13731232 | 7 | 3 | 0 | 1 |
| `rep2` | 0.61716193 | 6 | 5 | 2 | 2 |
| `rep3` | 0.87081246 | 7 | 5 | 1 | 1 |

- Median 600-frame ATE: `0.61716193 m`
- Reference point: the loop-enabled `room_depth` head-250 median above was `0.08255800 m`, so the longer `600`-frame run is much less stable.
- Inference from logs: the loop-correction stabilization path is active, not dead code. `rep2` and `rep3` both logged `TrackLocalMap: REJECTED - Recovery stabilization kept prior pose` and then forced reference refreshes after pending-loop-correction expiry, while `rep1` completed without stabilization rejections and had the best ATE.
- Net assessment: the loop stabilization changes help catch some bad post-loop handoffs, but they do not yet make `room_depth` reliable over `600` frames with loop closing enabled; two of the three runs still ended with large drift.

## Covisibility-Weighted Loop-Enabled Comparison (room_depth)

| Run | ATE (m) | Frames | Notes |
| --- | ---: | ---: | --- |
| `rep1` | 0.08620279 | 250 | Full run completed and wrote `250` poses |
| `rep2` | 0.08695826 | 250 | Loop edge confidence logged with `weight=7`; pose graph reported `covisibility_edges=424` and `loop_edges=1` |
| `rep3` | 0.07599828 | 250 | Loop edge confidence logged with `weight=7` |
| `median` | 0.08620279 | 250 | `+0.00364479 m` vs previous loop-enabled median `0.08255800` |

- Generated: `2026-04-14T10:06:11+09:00`
- SimpleVisualSLAM commit: `a32d904f66dbb77419ebc66529f6d5b1d6748302`
- Build: `build_codex`
- Command: `build_codex/run_mono --tum data/tum/rgbd_dataset_freiburg1_room --depth --max-frames 250 --no-viz data/ORBvoc.txt`
- Covisibility-weight verification: run logs include `LoopClosing: loop edge confidence ... weight=7`, confirming weighted loop-edge insertion is active with loop closing enabled on this scenario.
- Comparison to the prior loop-enabled `room_depth` median (`0.08255800`): this 3-run sample is worse by `0.00364479 m`, so the new weighting did not improve the median on this check.

## Metric Depth (DL) Results

| Run | Frames | ATE (m) | Notes |
| --- | ---: | ---: | --- |
| `room` | 50 | 0.01645454 | Loaded `models/depth_anything_v2_small.onnx`; finished with `50` poses |
| `room` | 250 | 0.38429766 | Loaded `models/depth_anything_v2_small.onnx`; finished with `250` poses; loop candidates were found but `computeSim3` rejected them |

- Generated: `2026-04-14T10:06:11+09:00`
- SimpleVisualSLAM commit: `a32d904f66dbb77419ebc66529f6d5b1d6748302`
- Build: `build_codex3`
- Build flags: `-DBUILD_TESTS=ON -DUSE_DEPTH_DL=ON`
- Command: `build_codex3/run_mono --tum data/tum/rgbd_dataset_freiburg1_room --metric-depth-model models/depth_anything_v2_small.onnx --max-frames {50,250} --no-viz`
- Model-load verification: logs show `MetricDepthEstimator: Loaded model from models/depth_anything_v2_small.onnx`.
- Loop-closing note: these metric-depth runs also loaded `data/ORBvoc.txt` and started `LoopClosing`; in the `250`-frame run, loop candidates at `KF 106` and `KF 109` were rejected by `computeSim3`, so no loop correction was applied within the measured window.
- Outcome: metric depth is functional on `room`, but the `250`-frame ATE (`0.38429766 m`) is much worse than the sensor-depth loop-enabled `room_depth` median measured above (`0.08620279 m`).
