# Reference Keyframe Decisions

## Adopted

- The shared `ReferenceKeyframePolicy` contract lives in `src/core/reference_keyframe_policy.h` and is now the boundary for reference-keyframe experiments.
- The runtime keeps `heuristic` as the default policy inside tracking. This preserves the current behavior while making the decision swappable.
- Every future policy trial must ship with an update to `experiments/reference_keyframe/scenarios.csv` and regenerated docs.

## Current Comparison Outcome

- Top curated-corpus accuracy tie: `score` (0.929), `pipeline` (0.929).
- The runtime default stays `heuristic` as the core baseline while experiments compete under the shared contract. Current curated counters: `fp=2`, `fn=0`.
- `pipeline` is the leading experimental candidate when latency matters (`23.76` ns/eval), but it still misses `room_mono_confident_refresh`.
- `score` stays as the conservative experiment with curated counters `fp=0`, `fn=1`.
- No experimental policy is promoted into `core` yet. The gate for adoption is replaying the same policy on real TUM/EuRoC traces, not just the curated scenario corpus.

## Real Trace Status

- Real-trace mean APE with `--repro-eval` enabled: `heuristic=0.099`, `score=0.070`, `pipeline=0.107`.
- Mode winners on the current bounded replay corpus: `depth=heuristic`, `depth_accel=score`, `mono=score`.
- The runtime default should only change if one policy wins on both curated scenarios and the bounded real-trace corpus.

## Mono Stability Status

- Mono repeat-2 mean/std APE: `heuristic=0.177±0.184`, `score=0.163±0.139`, `pipeline=0.125±0.080`.
- Worst mono variance is `room_mono_head/heuristic` with `std=0.146`. That makes repeat-based comparison mandatory for mono policy changes.

## Repro Eval Status

- Mono repeat-2 with `--repro-eval`: `heuristic=0.130±0.078`, `score=0.142±0.105`, `pipeline=0.113±0.061`.
- Worst repro mono variance is `room_mono_head/score` with `std=0.055`.
- Repro mode is an evaluation tool, not a runtime default: it exists to separate policy quality from async scheduling noise.

## Full Replay Stability Gate

- Full repeat-2 with `--repro-eval`: `heuristic=0.078±0.085`, `score=0.074±0.064`, `pipeline=0.083±0.088`.
- Repeat-2 mode winners on the full bounded corpus: `depth=score`, `depth_accel=heuristic`, `mono=score`.
- Worst full-corpus variance is `room_mono_head/heuristic` with `std=0.064`.
- No single policy dominates every mode under repeat replay, so the default stays in `core` and the experiments remain policy candidates instead of migrations.

## Room Hotspot Status

- Room-only repeat-2 with `--repro-eval`: `heuristic=0.146±0.088`, `score=0.164±0.132`, `pipeline=0.137±0.081`.
- Room-only mode winners: `depth_accel=score`, `mono=pipeline`.
- Worst room-only variance is `room_mono_head/score` with `std=0.115`.
- The room hotspot still does not justify a universal migration. It shows that `pipeline` can win locally on room windows while the full-corpus repeat gate still favors `score` overall.

## Rejected For Now

- Broad abstract refactors of `tracking` are still off the table. Only decision seams with comparable inputs and metrics graduate into `core`.
- A single canonical implementation is also off the table. `score` and `pipeline` stay intentionally discardable under `src/experiments/`.
