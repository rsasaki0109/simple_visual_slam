# EuRoC Test Results

Date: 2026-04-14

## Dataset acquisition

Attempted requested download:

```bash
mkdir -p data/euroc
wget -O /tmp/MH_01_easy.zip "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip"
```

Observed result:

- The legacy `robotics.ethz.ch` host resolved and accepted the TCP connection, but the transfer stalled with `/tmp/MH_01_easy.zip` remaining at `0` bytes.
- The newer ETH ASL datasets page was reachable, but as of 2026-04-14 it redirects to the moved datasets index and the EuRoC entry points to the ETH Research Collection DOI (`https://doi.org/10.3929/ethz-b-000690084`).
- The current official download exposed there is a combined `Machine Hall Datasets (ZIP, 12096.15 MB)` bundle rather than a direct `MH_01_easy.zip` asset.

Per the task rule, I used a synthetic EuRoC-style fallback dataset.

## Synthetic fallback dataset

Created:

- `data/euroc/test_seq/mav0/cam0/data/` with 10 grayscale `.pgm` frames
- `data/euroc/test_seq/mav0/cam0/data.csv`
- `data/euroc/test_seq/mav0/cam0/sensor.yaml`
- `data/euroc/test_seq/mav0/cam1/data/` with matching right-camera frames shifted by 8 px
- `data/euroc/test_seq/mav0/cam1/data.csv`
- `data/euroc/test_seq/mav0/cam1/sensor.yaml`

Notes:

- Image size: `752x480`
- Timestamps: `1000000000` to `1450000000` ns in 50 ms steps
- Stereo baseline in `sensor.yaml`: `0.110074 m`
- The run used `config/examples/euroc_mh01.json`

## Build

Executed:

```bash
cmake -S . -B build_codex -G Ninja -DBUILD_TESTS=ON && cmake --build build_codex -j$(nproc)
```

Result:

- Success
- `run_mono` rebuilt in `build_codex/run_mono`
- Build emitted one transient Ninja warning: `ninja: warning: premature end of file; recovering`

## EuRoC runner path caveat

The user-provided command form passes the inner `mav0` directory:

```bash
build_codex/run_mono --euroc data/euroc/test_seq/mav0 ...
```

In this repository, `EurocDataset` appends `/mav0/cam0/...` internally, so the command above fails with:

```text
Reason: data.csv not found: data/euroc/test_seq/mav0/mav0/cam0/data.csv
```

Successful runs therefore used the sequence root:

```bash
build_codex/run_mono --euroc data/euroc/test_seq ...
```

## Mono result

Executed:

```bash
build_codex/run_mono \
  --euroc data/euroc/test_seq \
  --euroc-camera-config config/examples/euroc_mh01.json \
  --max-frames 100 \
  --no-viz \
  --run-summary-json eval/euroc_mono_summary.json
```

Result:

- Exit code: `0`
- Processed frames: `10`
- Skipped frames: `0`
- Final tracking state: `2` (`OK`)
- Keyframes: `3`
- Landmarks: `277`
- Map saved: `true`

Summary file:

- `eval/euroc_mono_summary.json`

## Stereo result

Executed:

```bash
build_codex/run_mono \
  --euroc data/euroc/test_seq \
  --euroc-camera-config config/examples/euroc_mh01.json \
  --stereo \
  --max-frames 100 \
  --no-viz \
  --run-summary-json eval/euroc_stereo_summary.json
```

Result:

- Exit code: `0`
- Stereo mode reported: enabled
- Stereo depth estimation reported: enabled (`baseline=0.110074 m`)
- Processed frames: `10`
- Skipped frames: `0`
- Final tracking state: `2` (`OK`)
- Keyframes: `4`
- Landmarks: `1223`
- Map saved: `true`

Summary file:

- `eval/euroc_stereo_summary.json`

## Conclusion

- The real MH01 per-sequence archive could not be obtained from this environment.
- The synthetic EuRoC-style fallback dataset is valid for this loader and both mono and stereo execution paths complete successfully.
- Stereo mode exercised the EuRoC stereo code path and metric stereo depth initialization successfully.
