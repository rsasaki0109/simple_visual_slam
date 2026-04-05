# Reference Keyframe Decisions

## Adopted

- The shared `ReferenceKeyframePolicy` contract lives in `src/core/reference_keyframe_policy.h` and is now the boundary for reference-keyframe experiments.
- The runtime keeps `heuristic` as the default policy inside tracking. This preserves the current behavior while making the decision swappable.
- Every future policy trial must ship with an update to `experiments/reference_keyframe/scenarios.csv` and regenerated docs.

## Current Comparison Outcome

- Top curated-corpus accuracy tie: `score` (0.929), `pipeline` (0.929).
- The runtime default stays `heuristic` as the core baseline while experiments compete under the shared contract. Current curated counters: `fp=2`, `fn=0`.
- `pipeline` is the leading experimental candidate when latency matters (`20.59` ns/eval), but it still misses `room_mono_confident_refresh`.
- `score` stays as the conservative experiment with curated counters `fp=0`, `fn=1`.
- No experimental policy is promoted into `core` yet. The gate for adoption is replaying the same policy on real TUM/EuRoC traces, not just the curated scenario corpus.

## Real Trace Status

- Real-trace replay has not been generated yet. Run `bash scripts/eval_reference_policies.sh` before changing the default policy.

## Full Replay Stability Gate

- Full repeat-2 with `--repro-eval`: `heuristic=0.092±0.079`, `score=0.093±0.080`, `pipeline=0.107±0.117`.
- Repeat-2 mode winners on the full bounded corpus: `depth=heuristic`, `depth_accel=score`, `mono=score`.
- Worst full-corpus variance is `room_mono_head/score` with `std=0.080`.
- No single policy dominates every mode under repeat replay, so the default stays in `core` and the experiments remain policy candidates instead of migrations.

## Room Hotspot Status

- Room-only repeat-2 with `--repro-eval`: `heuristic=0.161±0.104`, `score=0.160±0.101`, `pipeline=0.160±0.105`.
- Room-only mode winners: `depth_accel=score`, `mono=pipeline`.
- Worst room-only variance is `room_mono_head/pipeline` with `std=0.107`.
- The room hotspot still does not justify a universal migration. It shows that `pipeline` can win locally on room windows while the full-corpus repeat gate still favors `score` overall.

## Rejected For Now

- Broad abstract refactors of `tracking` are still off the table. Only decision seams with comparable inputs and metrics graduate into `core`.
- A single canonical implementation is also off the table. `score` and `pipeline` stay intentionally discardable under `src/experiments/`.
