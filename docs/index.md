# Reference Keyframe Policy Summary

This is the GitHub-friendly landing page for the reference-keyframe experiment track.
It summarizes the current conclusion and links to the detailed experiment artifacts.

## Current Status

- Curated corpus top accuracy: `score` (0.929), `pipeline` (0.929).
- Bounded real-trace replay winner: `score` with mean APE `0.000`.
- Full repeat-2 replay winner: `score` with mean/std `0.093±0.080`.
- Runtime default is still `heuristic` because no single policy dominates every mode under repeat replay.

## Mode Snapshot

- Single-run bounded replay mode winners: ``.
- Repeat-2 bounded replay mode winners: `depth=heuristic, depth_accel=score, mono=score`.
- Room-only hotspot winners: `depth_accel=score, mono=pipeline`.

## Read Next

- [Detailed experiment tables](experiments.md)
- [Current decisions and adoption criteria](decisions.md)
- [Minimal interface that survived the experiments](interfaces.md)

## Reproduce

```bash
bash scripts/eval_reference_policies.sh --repeat 1
bash scripts/eval_reference_policies.sh --repeat 2 --output eval_results/reference_keyframe_policy/real_trace_metrics_repeat2.csv
bash scripts/eval_reference_policies.sh --repeat 2 --corpus experiments/reference_keyframe/room_focus_corpus.tsv --output eval_results/reference_keyframe_policy/room_focus_repeat2.csv
./scripts/update_reference_policy_docs.py
```
