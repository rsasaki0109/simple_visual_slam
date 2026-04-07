#!/usr/bin/env python3
"""Research comparison matrix: methods × chosen TUM windows, mean ATE (evo_ape Sim(3)), mean rank.

This is an internal / paper-style harness on TUM RGB-D snippets—not a KITTI odometry benchmark substitute
(KITTI uses different sensors, trajectories, and error definitions; chasing that head-on shrinks what you can evaluate fairly).

Requires: built run_mono, evo_ape on PATH, datasets under data/tum/ per eval/leaderboard_suite.json.

Example:
  python3 scripts/build_leaderboard.py --build build --output eval_results/leaderboard.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_lib  # noqa: E402

ROOT = eval_lib.ROOT
DEFAULT_SUITE = ROOT / "eval" / "leaderboard_suite.json"


def merge_run_gate(seq: dict, method: dict) -> dict:
    keys = (
        "tum_sequence",
        "skip_frames",
        "max_frames",
        "repro_eval",
        "reference_policy",
        "use_depth",
        "use_accel",
    )
    out: dict = {}
    for k in keys:
        if k in seq:
            out[k] = seq[k]
    for k in keys:
        if k in method:
            out[k] = method[k]
    out.setdefault("repro_eval", True)
    out.setdefault("skip_frames", 0)
    out.setdefault("use_depth", False)
    out.setdefault("use_accel", False)
    if "tum_sequence" not in out or "max_frames" not in out or "reference_policy" not in out:
        raise ValueError("incomplete sequence/method definition after merge")
    return out


def mean_tie_ranks(pairs: list[tuple[str, float]]) -> dict[str, float]:
    """Lower ATE is better. Ties share the average rank (1-based)."""
    sorted_m = sorted(pairs, key=lambda x: x[1])
    n = len(sorted_m)
    ranks: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        val = sorted_m[i][1]
        while j < n and sorted_m[j][1] == val:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[sorted_m[k][0]] = avg
        i = j
    return ranks


def format_cell(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, default=ROOT / "build", help="CMake build directory (run_mono cwd)")
    ap.add_argument("--data-tum", type=Path, default=ROOT / "data" / "tum")
    ap.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--output", type=Path, default=ROOT / "eval_results" / "leaderboard.md")
    ap.add_argument("--json-out", type=Path, default=None, help="Also write machine-readable results")
    ap.add_argument("--quiet", action="store_true", help="Suppress run_mono stdout/stderr")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Load suite and print planned runs only (no SLAM / evo_ape)",
    )
    args = ap.parse_args()

    suite = json.loads(args.suite.read_text())
    sequences = suite["sequences"]
    methods = suite["methods"]
    extra = suite.get("evo_ape_extra_args") or []
    build_dir = args.build.resolve()
    data_tum = args.data_tum.resolve()

    seq_ids = [s["id"] for s in sequences]
    method_ids = [m["id"] for m in methods]

    if args.dry_run:
        n = len(methods) * len(sequences)
        sys.stdout.write(
            f"[dry-run] suite={args.suite}\n"
            f"  sequences ({len(sequences)}): {', '.join(seq_ids)}\n"
            f"  methods ({len(methods)}): {', '.join(method_ids)}\n"
            f"  total SLAM runs: {n}\n"
            f"  evo_ape extra args: {extra}\n"
        )
        sys.stdout.write("Sample merged gate (first method × first sequence):\n")
        g = merge_run_gate(sequences[0], methods[0])
        sys.stdout.write(f"  {json.dumps(g, indent=2)}\n")
        return

    matrix: dict[str, dict[str, float | None]] = {m: {s: None for s in seq_ids} for m in method_ids}

    total = len(methods) * len(sequences)
    done = 0
    for method in methods:
        mid = method["id"]
        for seq in sequences:
            sid = seq["id"]
            done += 1
            gate = merge_run_gate(seq, method)
            tum_dir = data_tum / gate["tum_sequence"]
            gt = tum_dir / "groundtruth.txt"
            if not tum_dir.is_dir() or not gt.is_file():
                sys.stderr.write(f"[{done}/{total}] SKIP {mid}/{sid}: missing data under {tum_dir}\n")
                continue

            sys.stderr.write(f"[{done}/{total}] RUN {mid} × {sid} …\n")
            eval_lib.clean_traj_artifacts(build_dir)
            eval_lib.run_slam(build_dir, tum_dir, gate, quiet=args.quiet)
            traj = build_dir / "trajectory.txt"
            if not traj.is_file() or traj.stat().st_size == 0:
                sys.stderr.write(f"  no trajectory for {mid}/{sid}\n")
                continue
            ate = eval_lib.evo_mean_ape(gt, traj, extra, die_on_error=False)
            matrix[mid][sid] = ate
            if ate is not None:
                sys.stderr.write(f"  mean ATE = {ate:.6f} m\n")

    # Mean ATE per method (finite sequences only)
    mean_ates: dict[str, float] = {}
    for mid in method_ids:
        vals = [matrix[mid][s] for s in seq_ids if matrix[mid][s] is not None]
        mean_ates[mid] = statistics.mean(vals) if vals else float("inf")

    # Mean rank per method (average of tie-aware ranks per sequence)
    mean_ranks: dict[str, list[float]] = {m: [] for m in method_ids}
    for sid in seq_ids:
        pairs = [(m, matrix[m][sid]) for m in method_ids]
        pairs_f = [(m, v) for m, v in pairs if v is not None]
        if len(pairs_f) < 2:
            continue
        ranks = mean_tie_ranks(pairs_f)
        for m, r in ranks.items():
            mean_ranks[m].append(r)

    mean_rank_val = {m: (statistics.mean(mean_ranks[m]) if mean_ranks[m] else float("inf")) for m in method_ids}

    # Sort methods: primary mean ATE, secondary mean rank
    ordered = sorted(method_ids, key=lambda m: (mean_ates[m], mean_rank_val[m]))

    # Markdown
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# SimpleVisualSLAM — research comparison table (TUM)")
    lines.append("")
    lines.append(f"Generated: `{datetime.now(timezone.utc).isoformat()}` (suite `{args.suite.name}`)")
    lines.append("")
    lines.append(
        "**Scope:** curated TUM windows for ablations and policy comparison—not comparable to the KITTI Vision "
        "Benchmark leaderboard (different data, metrics, and community protocol)."
    )
    lines.append("")
    lines.append("Metric: **mean ATE (m)** after Sim(3) alignment (--align --correct_scale), same flags as `eval/regression_baselines.json`.")
    lines.append("Lower is better. **Mean rank** averages per-sequence ranks (1 = best on that sequence).")
    lines.append("")
    header = "| Rank | Method | " + " | ".join(seq_ids) + " | Mean ATE | Mean rank |"
    sep = "| --- | --- | " + " | ".join(["---"] * len(seq_ids)) + " | --- | --- |"
    lines.append(header)
    lines.append(sep)
    for rank_idx, mid in enumerate(ordered, start=1):
        cells = [format_cell(matrix[mid][s]) for s in seq_ids]
        ma = mean_ates[mid]
        mr = mean_rank_val[mid]
        ma_s = "—" if ma == float("inf") else f"{ma:.4f}"
        mr_s = "—" if mr == float("inf") else f"{mr:.2f}"
        lines.append(f"| {rank_idx} | `{mid}` | " + " | ".join(cells) + f" | {ma_s} | {mr_s} |")
    lines.append("")
    with args.output.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    sys.stdout.write(f"Wrote {args.output}\n")

    if args.json_out:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "suite_path": str(args.suite),
            "ordered_methods": ordered,
            "mean_ate_m": {m: (None if mean_ates[m] == float("inf") else mean_ates[m]) for m in method_ids},
            "mean_rank": {m: (None if mean_rank_val[m] == float("inf") else mean_rank_val[m]) for m in method_ids},
            "ate_m": matrix,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        sys.stdout.write(f"Wrote {args.json_out}\n")


if __name__ == "__main__":
    main()
