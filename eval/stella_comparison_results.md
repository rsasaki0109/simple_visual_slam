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
