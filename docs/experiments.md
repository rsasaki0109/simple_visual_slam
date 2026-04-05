# Reference Keyframe Experiments

Generated at: `2026-04-06T12:06:09.889672+00:00`

## Problem

Tracking currently decides whether a newly inserted keyframe should immediately become the reference anchor.
This pilot turns that decision into an experiment surface: one shared contract, one shared scenario corpus, and multiple competing policies.

## Shared Inputs

- `tracked_features`
- `detected_keypoints`
- `candidate_landmarks`
- `frames_since_reference`
- `lost_frames`
- `has_depth`
- `has_accel`

Scenario corpus: `experiments/reference_keyframe/scenarios.csv` with `14` comparable cases.

## Runtime Results

| Policy | Status | Philosophy | Accuracy | Precision | Recall | Promote Rate | Mean Confidence | Mean Eval ns |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic | core | imperative-thresholds | 0.857 | 0.818 | 1.000 | 0.786 | 0.794 | 24.47 |
| score | experiment | weighted-score | 0.929 | 1.000 | 0.889 | 0.571 | 0.784 | 29.48 |
| pipeline | experiment | staged-gates | 0.929 | 1.000 | 0.889 | 0.571 | 0.809 | 20.59 |

## Static Proxies

Readability and extensibility are heuristic scores generated from code size, branch count, named constants, and helper-function count.

| Policy | Non-comment LOC | Branch Points | Helper Functions | Named Constants | Readability | Extensibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heuristic | 56 | 2 | 2 | 2 | 3.59 | 3.04 |
| score | 86 | 7 | 7 | 9 | 1.83 | 5.00 |
| pipeline | 107 | 11 | 5 | 8 | 1.00 | 4.36 |

## Mismatch Hotspots

| Policy | Mismatched Scenarios |
| --- | --- |
| heuristic | room_mono_thin_map_support, room_depth_accel_thin_support |
| score | room_depth_after_minor_loss |
| pipeline | room_mono_confident_refresh |

## Full Replay Stability

This follow-up replays the full bounded real-trace corpus with `repeat 2` and `--repro-eval`.

| Policy | Runs | Mean APE | Std APE |
| --- | ---: | ---: | ---: |
| heuristic | 26 | 0.092 | 0.079 |
| score | 25 | 0.093 | 0.080 |
| pipeline | 26 | 0.107 | 0.117 |

| Mode | Policy | Runs | Mean APE | Std APE |
| --- | --- | ---: | ---: | ---: |
| depth | heuristic | 10 | 0.053 | 0.036 |
| depth | pipeline | 10 | 0.053 | 0.038 |
| depth | score | 9 | 0.056 | 0.033 |
| depth_accel | heuristic | 6 | 0.058 | 0.039 |
| depth_accel | pipeline | 6 | 0.058 | 0.041 |
| depth_accel | score | 6 | 0.057 | 0.037 |
| mono | heuristic | 10 | 0.151 | 0.091 |
| mono | pipeline | 10 | 0.191 | 0.148 |
| mono | score | 10 | 0.148 | 0.097 |

## Room Focus Follow-Up

This hotspot follow-up replays only `rgbd_dataset_freiburg1_room` windows for `mono` and `depth_accel` with `--repro-eval` enabled.

| Policy | Runs | Mean APE | Std APE |
| --- | ---: | ---: | ---: |
| heuristic | 49 | 0.161 | 0.104 |
| score | 50 | 0.160 | 0.101 |
| pipeline | 49 | 0.160 | 0.105 |

| Mode | Policy | Runs | Mean APE | Std APE |
| --- | --- | ---: | ---: | ---: |
| depth_accel | heuristic | 25 | 0.078 | 0.018 |
| depth_accel | pipeline | 25 | 0.080 | 0.020 |
| depth_accel | score | 25 | 0.077 | 0.025 |
| mono | heuristic | 24 | 0.247 | 0.085 |
| mono | pipeline | 24 | 0.243 | 0.092 |
| mono | score | 25 | 0.244 | 0.076 |

| Case | Policy | Runs | Mean APE | Std APE |
| --- | --- | ---: | ---: | ---: |
| room_depth_accel_head | heuristic | 5 | 0.102 | 0.012 |
| room_depth_accel_head | pipeline | 5 | 0.104 | 0.012 |
| room_depth_accel_head | score | 5 | 0.094 | 0.013 |
| room_depth_accel_late | heuristic | 5 | 0.056 | 0.009 |
| room_depth_accel_late | pipeline | 5 | 0.055 | 0.007 |
| room_depth_accel_late | score | 5 | 0.061 | 0.005 |
| room_depth_accel_mid | heuristic | 5 | 0.087 | 0.008 |
| room_depth_accel_mid | pipeline | 5 | 0.075 | 0.004 |
| room_depth_accel_mid | score | 5 | 0.074 | 0.007 |
| room_depth_accel_recovery | heuristic | 5 | 0.078 | 0.004 |
| room_depth_accel_recovery | pipeline | 5 | 0.081 | 0.007 |
| room_depth_accel_recovery | score | 5 | 0.077 | 0.006 |
| room_depth_accel_tail | heuristic | 5 | 0.068 | 0.004 |
| room_depth_accel_tail | pipeline | 5 | 0.084 | 0.020 |
| room_depth_accel_tail | score | 5 | 0.079 | 0.047 |
| room_mono_head | heuristic | 5 | 0.358 | 0.072 |
| room_mono_head | pipeline | 5 | 0.352 | 0.107 |
| room_mono_head | score | 5 | 0.316 | 0.063 |
| room_mono_late | heuristic | 5 | 0.147 | 0.028 |
| room_mono_late | pipeline | 5 | 0.147 | 0.036 |
| room_mono_late | score | 5 | 0.139 | 0.012 |
| room_mono_mid | heuristic | 5 | 0.213 | 0.019 |
| room_mono_mid | pipeline | 4 | 0.225 | 0.010 |
| room_mono_mid | score | 5 | 0.229 | 0.028 |
| room_mono_recovery | heuristic | 5 | 0.267 | 0.054 |
| room_mono_recovery | pipeline | 5 | 0.242 | 0.067 |
| room_mono_recovery | score | 5 | 0.279 | 0.047 |
| room_mono_tail | heuristic | 4 | 0.251 | 0.040 |
| room_mono_tail | pipeline | 5 | 0.246 | 0.041 |
| room_mono_tail | score | 5 | 0.256 | 0.065 |
