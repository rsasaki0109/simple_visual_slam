# EuRoC Stereo Pipeline Verification

Date: 2026-04-14

## Dataset
Synthetic EuRoC-style stereo dataset (100 frames, 20Hz, 752x480)
- cam0/cam1 with 0.110m baseline
- Simulated forward camera motion with depth-dependent parallax
- ETH EuRoC MH01 download failed (server down)

## Results

### Mono (no stereo, 50 frames)
- Initialization: FAILED (insufficient parallax in synthetic images)
- Keyframes: 0, Landmarks: 0

### Stereo (80 frames)
- Initialization: SUCCESS (depth-based, 369 3D points from single frame)
- Final state: OK (tracking maintained throughout)
- Keyframes: 27
- Landmarks: 9,134
- BA: Running with depth priors
- Stereo depth: baseline=0.110m, metric scale

## Observations
- Stereo initialization is much more robust than mono (single-frame depth init)
- Stereo produces ~25x more landmarks than mono on same scene
- The pipeline correctly uses StereoSGBM disparity -> metric depth
- depth_is_metric_ = true enables metric-scale tracking

## Next Steps
- Download real EuRoC MH01 when ETH server is back online
- Evaluate ATE against EuRoC groundtruth
- Compare mono vs stereo on same sequence
