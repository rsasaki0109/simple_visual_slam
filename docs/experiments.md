# Reference Keyframe Experiments

Generated at: `2026-04-02T21:12:15.235328+00:00`

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
| heuristic | core | imperative-thresholds | 0.857 | 0.818 | 1.000 | 0.786 | 0.794 | 33.94 |
| score | experiment | weighted-score | 0.929 | 1.000 | 0.889 | 0.571 | 0.784 | 40.94 |
| pipeline | experiment | staged-gates | 0.929 | 1.000 | 0.889 | 0.571 | 0.809 | 23.76 |

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

## Real Trace Replay

The real-trace corpus replays fixed windows from TUM sequences with `--repro-eval` enabled so policies are compared on the same bounded input budget.

| Policy | Successful Runs | Mean APE | Mean RMSE |
| --- | ---: | ---: | ---: |
| heuristic | 10 | 0.099 | 0.111 |
| score | 10 | 0.070 | 0.082 |
| pipeline | 10 | 0.107 | 0.120 |

| Mode | Policy | Successful Runs | Mean APE | Mean RMSE |
| --- | --- | ---: | ---: | ---: |
| depth | heuristic | 4 | 0.040 | 0.045 |
| depth | score | 4 | 0.043 | 0.050 |
| depth | pipeline | 4 | 0.046 | 0.053 |
| depth_accel | heuristic | 2 | 0.036 | 0.040 |
| depth_accel | score | 2 | 0.031 | 0.035 |
| depth_accel | pipeline | 2 | 0.039 | 0.045 |
| mono | heuristic | 4 | 0.190 | 0.213 |
| mono | score | 4 | 0.117 | 0.138 |
| mono | pipeline | 4 | 0.200 | 0.225 |

| Case | Policy | Mode | Skip Frames | Max Frames | Mean APE | RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| room_depth_accel_late | heuristic | depth_accel | 500 | 250 | 0.062 | 0.068 |
| room_depth_accel_late | pipeline | depth_accel | 500 | 250 | 0.068 | 0.077 |
| room_depth_accel_late | score | depth_accel | 500 | 250 | 0.050 | 0.058 |
| room_depth_head | heuristic | depth | 0 | 250 | 0.079 | 0.087 |
| room_depth_head | pipeline | depth | 0 | 250 | 0.103 | 0.121 |
| room_depth_head | score | depth | 0 | 250 | 0.095 | 0.111 |
| room_depth_late | heuristic | depth | 500 | 250 | 0.058 | 0.065 |
| room_depth_late | pipeline | depth | 500 | 250 | 0.058 | 0.065 |
| room_depth_late | score | depth | 500 | 250 | 0.056 | 0.063 |
| room_mono_head | heuristic | mono | 0 | 250 | 0.507 | 0.546 |
| room_mono_head | pipeline | mono | 0 | 250 | 0.573 | 0.613 |
| room_mono_head | score | mono | 0 | 250 | 0.194 | 0.216 |
| room_mono_late | heuristic | mono | 500 | 250 | 0.141 | 0.169 |
| room_mono_late | pipeline | mono | 500 | 250 | 0.118 | 0.149 |
| room_mono_late | score | mono | 500 | 250 | 0.156 | 0.185 |
| xyz_depth_accel_head | heuristic | depth_accel | 0 | 250 | 0.011 | 0.012 |
| xyz_depth_accel_head | pipeline | depth_accel | 0 | 250 | 0.011 | 0.012 |
| xyz_depth_accel_head | score | depth_accel | 0 | 250 | 0.011 | 0.012 |
| xyz_depth_head | heuristic | depth | 0 | 250 | 0.011 | 0.013 |
| xyz_depth_head | pipeline | depth | 0 | 250 | 0.010 | 0.011 |
| xyz_depth_head | score | depth | 0 | 250 | 0.010 | 0.011 |
| xyz_depth_late | heuristic | depth | 300 | 250 | 0.013 | 0.015 |
| xyz_depth_late | pipeline | depth | 300 | 250 | 0.014 | 0.015 |
| xyz_depth_late | score | depth | 300 | 250 | 0.012 | 0.013 |
| xyz_mono_head | heuristic | mono | 0 | 250 | 0.040 | 0.053 |
| xyz_mono_head | pipeline | mono | 0 | 250 | 0.041 | 0.055 |
| xyz_mono_head | score | mono | 0 | 250 | 0.048 | 0.066 |
| xyz_mono_late | heuristic | mono | 300 | 250 | 0.072 | 0.084 |
| xyz_mono_late | pipeline | mono | 300 | 250 | 0.070 | 0.082 |
| xyz_mono_late | score | mono | 300 | 250 | 0.072 | 0.083 |

## Mono Stability Follow-Up

This follow-up replays the mono subset with `repeat 2` to expose run-to-run variance.

| Policy | Runs | Mean APE | Std APE |
| --- | ---: | ---: | ---: |
| heuristic | 8 | 0.177 | 0.184 |
| score | 8 | 0.163 | 0.139 |
| pipeline | 8 | 0.125 | 0.080 |

| Case | Policy | Runs | Mean APE | Std APE |
| --- | --- | ---: | ---: | ---: |
| room_mono_head | heuristic | 2 | 0.461 | 0.146 |
| room_mono_head | pipeline | 2 | 0.220 | 0.039 |
| room_mono_head | score | 2 | 0.394 | 0.016 |
| room_mono_late | heuristic | 2 | 0.144 | 0.011 |
| room_mono_late | pipeline | 2 | 0.179 | 0.007 |
| room_mono_late | score | 2 | 0.140 | 0.021 |
| xyz_mono_head | heuristic | 2 | 0.030 | 0.002 |
| xyz_mono_head | pipeline | 2 | 0.034 | 0.006 |
| xyz_mono_head | score | 2 | 0.039 | 0.006 |
| xyz_mono_late | heuristic | 2 | 0.072 | 0.004 |
| xyz_mono_late | pipeline | 2 | 0.065 | 0.003 |
| xyz_mono_late | score | 2 | 0.080 | 0.011 |

## Mono Stability With Repro Eval

This follow-up replays the same mono subset with `--repro-eval`, forcing synchronous local mapping and disabling loop closing.

| Policy | Runs | Mean APE | Std APE |
| --- | ---: | ---: | ---: |
| heuristic | 8 | 0.130 | 0.078 |
| score | 8 | 0.142 | 0.105 |
| pipeline | 8 | 0.113 | 0.061 |

| Case | Policy | Runs | Mean APE | Std APE |
| --- | --- | ---: | ---: | ---: |
| room_mono_head | heuristic | 2 | 0.246 | 0.014 |
| room_mono_head | pipeline | 2 | 0.206 | 0.009 |
| room_mono_head | score | 2 | 0.296 | 0.055 |
| room_mono_late | heuristic | 2 | 0.150 | 0.016 |
| room_mono_late | pipeline | 2 | 0.126 | 0.005 |
| room_mono_late | score | 2 | 0.165 | 0.011 |
| xyz_mono_head | heuristic | 2 | 0.043 | 0.003 |
| xyz_mono_head | pipeline | 2 | 0.045 | 0.002 |
| xyz_mono_head | score | 2 | 0.037 | 0.005 |
| xyz_mono_late | heuristic | 2 | 0.079 | 0.008 |
| xyz_mono_late | pipeline | 2 | 0.075 | 0.003 |
| xyz_mono_late | score | 2 | 0.070 | 0.000 |

## Full Replay Stability

This follow-up replays the full bounded real-trace corpus with `repeat 2` and `--repro-eval`.

| Policy | Runs | Mean APE | Std APE |
| --- | ---: | ---: | ---: |
| heuristic | 20 | 0.078 | 0.085 |
| score | 20 | 0.074 | 0.064 |
| pipeline | 20 | 0.083 | 0.088 |

| Mode | Policy | Runs | Mean APE | Std APE |
| --- | --- | ---: | ---: | ---: |
| depth | heuristic | 8 | 0.040 | 0.029 |
| depth | pipeline | 8 | 0.042 | 0.032 |
| depth | score | 8 | 0.039 | 0.029 |
| depth_accel | heuristic | 4 | 0.033 | 0.022 |
| depth_accel | pipeline | 4 | 0.034 | 0.023 |
| depth_accel | score | 4 | 0.042 | 0.031 |
| mono | heuristic | 8 | 0.138 | 0.104 |
| mono | pipeline | 8 | 0.148 | 0.106 |
| mono | score | 8 | 0.124 | 0.070 |

## Room Focus Follow-Up

This hotspot follow-up replays only `rgbd_dataset_freiburg1_room` windows for `mono` and `depth_accel` with `repeat 2` and `--repro-eval`.

| Policy | Runs | Mean APE | Std APE |
| --- | ---: | ---: | ---: |
| heuristic | 12 | 0.146 | 0.088 |
| score | 12 | 0.164 | 0.132 |
| pipeline | 12 | 0.137 | 0.081 |

| Mode | Policy | Runs | Mean APE | Std APE |
| --- | --- | ---: | ---: | ---: |
| depth_accel | heuristic | 6 | 0.070 | 0.008 |
| depth_accel | pipeline | 6 | 0.073 | 0.011 |
| depth_accel | score | 6 | 0.069 | 0.018 |
| mono | heuristic | 6 | 0.222 | 0.061 |
| mono | pipeline | 6 | 0.201 | 0.070 |
| mono | score | 6 | 0.259 | 0.128 |

| Case | Policy | Runs | Mean APE | Std APE |
| --- | --- | ---: | ---: | ---: |
| room_depth_accel_head | heuristic | 2 | 0.077 | 0.000 |
| room_depth_accel_head | pipeline | 2 | 0.084 | 0.006 |
| room_depth_accel_head | score | 2 | 0.092 | 0.009 |
| room_depth_accel_late | heuristic | 2 | 0.070 | 0.000 |
| room_depth_accel_late | pipeline | 2 | 0.070 | 0.000 |
| room_depth_accel_late | score | 2 | 0.056 | 0.005 |
| room_depth_accel_mid | heuristic | 2 | 0.062 | 0.009 |
| room_depth_accel_mid | pipeline | 2 | 0.065 | 0.011 |
| room_depth_accel_mid | score | 2 | 0.058 | 0.009 |
| room_mono_head | heuristic | 2 | 0.294 | 0.034 |
| room_mono_head | pipeline | 2 | 0.266 | 0.077 |
| room_mono_head | score | 2 | 0.409 | 0.115 |
| room_mono_late | heuristic | 2 | 0.156 | 0.020 |
| room_mono_late | pipeline | 2 | 0.142 | 0.028 |
| room_mono_late | score | 2 | 0.153 | 0.005 |
| room_mono_mid | heuristic | 2 | 0.216 | 0.000 |
| room_mono_mid | pipeline | 2 | 0.196 | 0.004 |
| room_mono_mid | score | 2 | 0.215 | 0.004 |
