#!/usr/bin/env python3

import csv
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "build"
SCENARIO_FILE = ROOT / "experiments" / "reference_keyframe" / "scenarios.csv"
OUTPUT_DIR = ROOT / "eval_results" / "reference_keyframe_policy"
METRICS_FILE = OUTPUT_DIR / "metrics.csv"
DECISIONS_FILE = OUTPUT_DIR / "decisions.csv"
STATIC_FILE = OUTPUT_DIR / "static_metrics.csv"
REAL_TRACE_FILE = OUTPUT_DIR / "real_trace_metrics.csv"
MONO_STABILITY_FILE = OUTPUT_DIR / "real_trace_metrics_mono_repeat2.csv"
MONO_REPRO_STABILITY_FILE = OUTPUT_DIR / "real_trace_metrics_mono_repeat2_repro.csv"
FULL_REPRO_STABILITY_FILE = OUTPUT_DIR / "real_trace_metrics_repeat2.csv"
ROOM_FOCUS_STABILITY_FILE = OUTPUT_DIR / "room_focus_repeat5.csv"
DOCS_DIR = ROOT / "docs"

POLICIES = {
    "heuristic": {
        "philosophy": "imperative-thresholds",
        "files": [
            ROOT / "src" / "core" / "heuristic_reference_keyframe_policy.h",
            ROOT / "src" / "core" / "heuristic_reference_keyframe_policy.cc",
        ],
        "status": "core",
    },
    "score": {
        "philosophy": "weighted-score",
        "files": [
            ROOT / "src" / "experiments" / "reference_keyframe" / "score_reference_keyframe_policy.h",
            ROOT / "src" / "experiments" / "reference_keyframe" / "score_reference_keyframe_policy.cc",
        ],
        "status": "experiment",
    },
    "pipeline": {
        "philosophy": "staged-gates",
        "files": [
            ROOT / "src" / "experiments" / "reference_keyframe" / "pipeline_reference_keyframe_policy.h",
            ROOT / "src" / "experiments" / "reference_keyframe" / "pipeline_reference_keyframe_policy.cc",
        ],
        "status": "experiment",
    },
}


def run(cmd):
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def non_comment_loc(text):
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith("/*") or stripped.startswith("*") or stripped.startswith("*/"):
            continue
        count += 1
    return count


def branch_points(text):
    patterns = [
        r"\bif\s*\(",
        r"\belse\s+if\s*\(",
        r"\bfor\s*\(",
        r"\bwhile\s*\(",
        r"\bswitch\s*\(",
        r"\?",
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def helper_functions(text):
    names = re.findall(r"::([A-Za-z0-9_]+)\s*\(", text)
    return len(
        {
            name
            for name in names
            if name not in {"name", "philosophy", "evaluate"}
        }
    )


def named_constants(text):
    return len(set(re.findall(r"\bk[A-Z][A-Za-z0-9_]*\b", text)))


def include_count(text):
    return len(re.findall(r'^\s*#include\s+', text, flags=re.MULTILINE))


def clamp(value, lower=1.0, upper=5.0):
    return max(lower, min(upper, value))


def collect_static_metrics():
    rows = []
    for policy_name, metadata in POLICIES.items():
        joined = "\n".join(path.read_text() for path in metadata["files"])
        loc = non_comment_loc(joined)
        branches = branch_points(joined)
        helpers = helper_functions(joined)
        constants = named_constants(joined)
        includes = include_count(joined)
        readability = clamp(5.6 - loc / 35.0 - branches * 0.18 - max(0, includes - 3) * 0.05)
        extensibility = clamp(2.2 + helpers * 0.35 + constants * 0.12 - branches * 0.05)
        rows.append(
            {
                "policy": policy_name,
                "philosophy": metadata["philosophy"],
                "status": metadata["status"],
                "non_comment_loc": loc,
                "branch_points": branches,
                "helper_functions": helpers,
                "named_constants": constants,
                "include_count": includes,
                "readability_score": f"{readability:.2f}",
                "extensibility_score": f"{extensibility:.2f}",
            }
        )
    with STATIC_FILE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_joined_results():
    runtime_rows = {row["policy"]: row for row in read_csv(METRICS_FILE)}
    static_rows = {row["policy"]: row for row in read_csv(STATIC_FILE)}
    joined = []
    for policy in POLICIES:
        merged = {}
        merged.update(runtime_rows[policy])
        merged.update(static_rows[policy])
        joined.append(merged)
    return joined


def best_candidate(rows):
    return max(
        rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["recall"]),
            -float(row["mean_eval_ns"]),
        ),
    )


def best_accuracy_rows(rows):
    best_accuracy = max(float(row["accuracy"]) for row in rows)
    return [row for row in rows if float(row["accuracy"]) == best_accuracy]


def curated_accuracy_line(rows):
    best_rows = best_accuracy_rows(rows)
    if len(best_rows) == 1:
        row = best_rows[0]
        return f"- Top curated-corpus accuracy: `{row['policy']}` ({float(row['accuracy']):.3f})."
    tie_line = ", ".join(
        f"`{row['policy']}` ({float(row['accuracy']):.3f})" for row in best_rows
    )
    return f"- Top curated-corpus accuracy tie: {tie_line}."


def mismatch_summary():
    rows = read_csv(DECISIONS_FILE)
    summary = {}
    for row in rows:
        summary.setdefault(row["policy"], [])
        if row["match"] == "0":
            summary[row["policy"]].append(row["scenario"])
    return summary


def load_real_trace_rows():
    if not REAL_TRACE_FILE.exists():
        return []
    rows = read_csv(REAL_TRACE_FILE)
    return [row for row in rows if row["status"] == "OK"]


def aggregate_real_trace_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["policy"], [])
        grouped[row["policy"]].append(row)

    summary = {}
    for policy, policy_rows in grouped.items():
        mean_values = [float(row["mean"]) for row in policy_rows]
        rmse_values = [float(row["rmse"]) for row in policy_rows]
        summary[policy] = {
            "successful_runs": len(policy_rows),
            "mean_ape": sum(mean_values) / len(mean_values),
            "mean_rmse": sum(rmse_values) / len(rmse_values),
        }
    return summary


def aggregate_rows_with_std(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, [])
        grouped[key].append(float(row["mean"]))

    summary = {}
    for key, values in grouped.items():
        mean_value = sum(values) / len(values)
        variance = sum(value * value for value in values) / len(values) - mean_value * mean_value
        summary[key] = {
            "mean_ape": mean_value,
            "std_ape": variance ** 0.5 if variance > 0.0 else 0.0,
            "runs": len(values),
        }
    return summary


def aggregate_real_trace_rows_by_mode(rows):
    grouped = {}
    for row in rows:
        key = (row["policy"], row["mode"])
        grouped.setdefault(key, [])
        grouped[key].append(row)

    summary = {}
    for key, policy_rows in grouped.items():
        mean_values = [float(row["mean"]) for row in policy_rows]
        rmse_values = [float(row["rmse"]) for row in policy_rows]
        summary[key] = {
            "successful_runs": len(policy_rows),
            "mean_ape": sum(mean_values) / len(mean_values),
            "mean_rmse": sum(rmse_values) / len(rmse_values),
        }
    return summary


def best_real_trace_policy_by_mode(rows):
    summary = aggregate_real_trace_rows_by_mode(rows)
    best = {}
    for mode in {row["mode"] for row in rows}:
        candidates = []
        for policy in POLICIES:
            aggregate = summary.get((policy, mode))
            if not aggregate:
                continue
            candidates.append((aggregate["mean_ape"], aggregate["mean_rmse"], policy))
        if candidates:
            candidates.sort()
            best[mode] = candidates[0][2]
    return best


def load_mono_stability_rows():
    if not MONO_STABILITY_FILE.exists():
        return []
    rows = read_csv(MONO_STABILITY_FILE)
    return [row for row in rows if row["status"] == "OK"]


def load_mono_repro_stability_rows():
    if not MONO_REPRO_STABILITY_FILE.exists():
        return []
    rows = read_csv(MONO_REPRO_STABILITY_FILE)
    return [row for row in rows if row["status"] == "OK"]


def load_full_repro_stability_rows():
    if not FULL_REPRO_STABILITY_FILE.exists():
        return []
    rows = read_csv(FULL_REPRO_STABILITY_FILE)
    return [row for row in rows if row["status"] == "OK"]


def load_room_focus_stability_rows():
    if not ROOM_FOCUS_STABILITY_FILE.exists():
        return []
    rows = read_csv(ROOM_FOCUS_STABILITY_FILE)
    return [row for row in rows if row["status"] == "OK"]


def describe_repro_mode(rows):
    values = {row.get("repro_eval", "") for row in rows}
    if values == {"1"}:
        return "with `--repro-eval` enabled"
    if values == {"0"}:
        return "without `--repro-eval`"
    return "with mixed replay settings"


def generate_experiments_doc(rows, scenarios):
    generated_at = datetime.now(timezone.utc).isoformat()
    mismatches = mismatch_summary()
    real_trace_rows = load_real_trace_rows()
    real_trace_summary = aggregate_real_trace_rows(real_trace_rows)
    real_trace_mode_summary = aggregate_real_trace_rows_by_mode(real_trace_rows)
    real_trace_repro_mode = describe_repro_mode(real_trace_rows) if real_trace_rows else ""
    mono_stability_rows = load_mono_stability_rows()
    mono_stability_summary = aggregate_rows_with_std(mono_stability_rows, ["policy"])
    mono_stability_case_summary = aggregate_rows_with_std(mono_stability_rows, ["case_id", "policy"])
    mono_repro_rows = load_mono_repro_stability_rows()
    mono_repro_summary = aggregate_rows_with_std(mono_repro_rows, ["policy"])
    mono_repro_case_summary = aggregate_rows_with_std(mono_repro_rows, ["case_id", "policy"])
    full_repro_rows = load_full_repro_stability_rows()
    full_repro_summary = aggregate_rows_with_std(full_repro_rows, ["policy"])
    full_repro_mode_summary = aggregate_rows_with_std(full_repro_rows, ["mode", "policy"])
    room_focus_rows = load_room_focus_stability_rows()
    room_focus_summary = aggregate_rows_with_std(room_focus_rows, ["policy"])
    room_focus_mode_summary = aggregate_rows_with_std(room_focus_rows, ["mode", "policy"])
    room_focus_case_summary = aggregate_rows_with_std(room_focus_rows, ["case_id", "policy"])
    lines = [
        "# Reference Keyframe Experiments",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Problem",
        "",
        "Tracking currently decides whether a newly inserted keyframe should immediately become the reference anchor.",
        "This pilot turns that decision into an experiment surface: one shared contract, one shared scenario corpus, and multiple competing policies.",
        "",
        "## Shared Inputs",
        "",
        "- `tracked_features`",
        "- `detected_keypoints`",
        "- `candidate_landmarks`",
        "- `frames_since_reference`",
        "- `lost_frames`",
        "- `has_depth`",
        "- `has_accel`",
        "",
        f"Scenario corpus: `{SCENARIO_FILE.relative_to(ROOT)}` with `{len(scenarios)}` comparable cases.",
        "",
        "## Runtime Results",
        "",
        "| Policy | Status | Philosophy | Accuracy | Precision | Recall | Promote Rate | Mean Confidence | Mean Eval ns |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {policy} | {status} | {philosophy} | {accuracy:.3f} | {precision:.3f} | {recall:.3f} | {promote_rate:.3f} | {mean_confidence:.3f} | {mean_eval_ns:.2f} |".format(
                policy=row["policy"],
                status=row["status"],
                philosophy=row["philosophy"],
                accuracy=float(row["accuracy"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                promote_rate=float(row["promote_rate"]),
                mean_confidence=float(row["mean_confidence"]),
                mean_eval_ns=float(row["mean_eval_ns"]),
            )
        )
    lines.extend(
        [
            "",
            "## Static Proxies",
            "",
            "Readability and extensibility are heuristic scores generated from code size, branch count, named constants, and helper-function count.",
            "",
            "| Policy | Non-comment LOC | Branch Points | Helper Functions | Named Constants | Readability | Extensibility |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {policy} | {loc} | {branches} | {helpers} | {constants} | {readability} | {extensibility} |".format(
                policy=row["policy"],
                loc=row["non_comment_loc"],
                branches=row["branch_points"],
                helpers=row["helper_functions"],
                constants=row["named_constants"],
                readability=row["readability_score"],
                extensibility=row["extensibility_score"],
            )
        )
    lines.extend(
        [
            "",
            "## Mismatch Hotspots",
            "",
            "| Policy | Mismatched Scenarios |",
            "| --- | --- |",
        ]
    )
    for row in rows:
        misses = mismatches.get(row["policy"], [])
        lines.append(
            f"| {row['policy']} | {', '.join(misses) if misses else 'none'} |"
        )
    if real_trace_rows:
        lines.extend(
            [
                "",
                "## Real Trace Replay",
                "",
                f"The real-trace corpus replays fixed windows from TUM sequences {real_trace_repro_mode} so policies are compared on the same bounded input budget.",
                "",
                "| Policy | Successful Runs | Mean APE | Mean RMSE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for policy in POLICIES:
            aggregate = real_trace_summary.get(policy)
            if not aggregate:
                continue
            lines.append(
                "| {policy} | {successful_runs} | {mean_ape:.3f} | {mean_rmse:.3f} |".format(
                    policy=policy,
                    successful_runs=aggregate["successful_runs"],
                    mean_ape=aggregate["mean_ape"],
                    mean_rmse=aggregate["mean_rmse"],
                )
            )
        lines.extend(
            [
                "",
                "| Mode | Policy | Successful Runs | Mean APE | Mean RMSE |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for mode in sorted({row["mode"] for row in real_trace_rows}):
            for policy in POLICIES:
                aggregate = real_trace_mode_summary.get((policy, mode))
                if not aggregate:
                    continue
                lines.append(
                    "| {mode} | {policy} | {successful_runs} | {mean_ape:.3f} | {mean_rmse:.3f} |".format(
                        mode=mode,
                        policy=policy,
                        successful_runs=aggregate["successful_runs"],
                        mean_ape=aggregate["mean_ape"],
                        mean_rmse=aggregate["mean_rmse"],
                    )
                )
        lines.extend(
            [
                "",
                "| Case | Policy | Mode | Skip Frames | Max Frames | Mean APE | RMSE |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in sorted(real_trace_rows, key=lambda item: (item["case_id"], item["policy"], int(item["run_idx"]))):
            lines.append(
                "| {case_id} | {policy} | {mode} | {skip_frames} | {max_frames} | {mean:.3f} | {rmse:.3f} |".format(
                    case_id=row["case_id"],
                    policy=row["policy"],
                    mode=row["mode"],
                    skip_frames=int(row["skip_frames"]),
                    max_frames=int(row["max_frames"]),
                    mean=float(row["mean"]),
                    rmse=float(row["rmse"]),
                )
            )
    if mono_stability_rows:
        lines.extend(
            [
                "",
                "## Mono Stability Follow-Up",
                "",
                "This follow-up replays the mono subset with `repeat 2` to expose run-to-run variance.",
                "",
                "| Policy | Runs | Mean APE | Std APE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for policy in POLICIES:
            aggregate = mono_stability_summary.get((policy,))
            if not aggregate:
                continue
            lines.append(
                "| {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
        lines.extend(
            [
                "",
                "| Case | Policy | Runs | Mean APE | Std APE |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for case_id, policy in sorted(mono_stability_case_summary):
            aggregate = mono_stability_case_summary[(case_id, policy)]
            lines.append(
                "| {case_id} | {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    case_id=case_id,
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
    if mono_repro_rows:
        lines.extend(
            [
                "",
                "## Mono Stability With Repro Eval",
                "",
                "This follow-up replays the same mono subset with `--repro-eval`, forcing synchronous local mapping and disabling loop closing.",
                "",
                "| Policy | Runs | Mean APE | Std APE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for policy in POLICIES:
            aggregate = mono_repro_summary.get((policy,))
            if not aggregate:
                continue
            lines.append(
                "| {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
        lines.extend(
            [
                "",
                "| Case | Policy | Runs | Mean APE | Std APE |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for case_id, policy in sorted(mono_repro_case_summary):
            aggregate = mono_repro_case_summary[(case_id, policy)]
            lines.append(
                "| {case_id} | {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    case_id=case_id,
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
    if full_repro_rows:
        lines.extend(
            [
                "",
                "## Full Replay Stability",
                "",
                "This follow-up replays the full bounded real-trace corpus with `repeat 2` and `--repro-eval`.",
                "",
                "| Policy | Runs | Mean APE | Std APE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for policy in POLICIES:
            aggregate = full_repro_summary.get((policy,))
            if not aggregate:
                continue
            lines.append(
                "| {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
        lines.extend(
            [
                "",
                "| Mode | Policy | Runs | Mean APE | Std APE |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for mode, policy in sorted(full_repro_mode_summary):
            aggregate = full_repro_mode_summary[(mode, policy)]
            lines.append(
                "| {mode} | {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    mode=mode,
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
    if room_focus_rows:
        lines.extend(
            [
                "",
                "## Room Focus Follow-Up",
                "",
                "This hotspot follow-up replays only `rgbd_dataset_freiburg1_room` windows for `mono` and `depth_accel` {repro}.".format(
                    repro=describe_repro_mode(room_focus_rows),
                ),
                "",
                "| Policy | Runs | Mean APE | Std APE |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for policy in POLICIES:
            aggregate = room_focus_summary.get((policy,))
            if not aggregate:
                continue
            lines.append(
                "| {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
        lines.extend(
            [
                "",
                "| Mode | Policy | Runs | Mean APE | Std APE |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for mode, policy in sorted(room_focus_mode_summary):
            aggregate = room_focus_mode_summary[(mode, policy)]
            lines.append(
                "| {mode} | {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    mode=mode,
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
        lines.extend(
            [
                "",
                "| Case | Policy | Runs | Mean APE | Std APE |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for case_id, policy in sorted(room_focus_case_summary):
            aggregate = room_focus_case_summary[(case_id, policy)]
            lines.append(
                "| {case_id} | {policy} | {runs} | {mean_ape:.3f} | {std_ape:.3f} |".format(
                    case_id=case_id,
                    policy=policy,
                    runs=aggregate["runs"],
                    mean_ape=aggregate["mean_ape"],
                    std_ape=aggregate["std_ape"],
                )
            )
    DOCS_DIR.joinpath("experiments.md").write_text("\n".join(lines) + "\n")


def generate_decisions_doc(rows):
    mismatches = mismatch_summary()
    real_trace_rows = load_real_trace_rows()
    real_trace_summary = aggregate_real_trace_rows(real_trace_rows)
    real_trace_mode_winners = best_real_trace_policy_by_mode(real_trace_rows)
    real_trace_repro_mode = describe_repro_mode(real_trace_rows) if real_trace_rows else ""
    mono_stability_rows = load_mono_stability_rows()
    mono_stability_summary = aggregate_rows_with_std(mono_stability_rows, ["policy"])
    mono_stability_case_summary = aggregate_rows_with_std(mono_stability_rows, ["case_id", "policy"])
    mono_repro_rows = load_mono_repro_stability_rows()
    mono_repro_summary = aggregate_rows_with_std(mono_repro_rows, ["policy"])
    mono_repro_case_summary = aggregate_rows_with_std(mono_repro_rows, ["case_id", "policy"])
    full_repro_rows = load_full_repro_stability_rows()
    full_repro_summary = aggregate_rows_with_std(full_repro_rows, ["policy"])
    full_repro_mode_summary = aggregate_rows_with_std(full_repro_rows, ["mode", "policy"])
    full_repro_case_summary = aggregate_rows_with_std(full_repro_rows, ["case_id", "policy"])
    room_focus_rows = load_room_focus_stability_rows()
    room_focus_summary = aggregate_rows_with_std(room_focus_rows, ["policy"])
    room_focus_mode_summary = aggregate_rows_with_std(room_focus_rows, ["mode", "policy"])
    room_focus_case_summary = aggregate_rows_with_std(room_focus_rows, ["case_id", "policy"])
    lines = [
        "# Reference Keyframe Decisions",
        "",
        "## Adopted",
        "",
        "- The shared `ReferenceKeyframePolicy` contract lives in `src/core/reference_keyframe_policy.h` and is now the boundary for reference-keyframe experiments.",
        "- The runtime keeps `heuristic` as the default policy inside tracking. This preserves the current behavior while making the decision swappable.",
        "- Every future policy trial must ship with an update to `experiments/reference_keyframe/scenarios.csv` and regenerated docs.",
        "",
        "## Current Comparison Outcome",
        "",
        curated_accuracy_line(rows),
        f"- The runtime default stays `heuristic` as the core baseline while experiments compete under the shared contract. Current curated counters: `fp={next(row['fp'] for row in rows if row['policy'] == 'heuristic')}`, `fn={next(row['fn'] for row in rows if row['policy'] == 'heuristic')}`.",
        f"- `pipeline` is the leading experimental candidate when latency matters (`{float(next(row['mean_eval_ns'] for row in rows if row['policy'] == 'pipeline')):.2f}` ns/eval), but it still misses `{', '.join(mismatches.get('pipeline', [])) or 'none'}`.",
        f"- `score` stays as the conservative experiment with curated counters `fp={next(row['fp'] for row in rows if row['policy'] == 'score')}`, `fn={next(row['fn'] for row in rows if row['policy'] == 'score')}`.",
        "- No experimental policy is promoted into `core` yet. The gate for adoption is replaying the same policy on real TUM/EuRoC traces, not just the curated scenario corpus.",
        "",
        "## Real Trace Status",
        "",
    ]
    if real_trace_summary:
        mode_winner_text = ", ".join(
            f"`{mode}={policy}`" for mode, policy in sorted(real_trace_mode_winners.items())
        )
        lines.extend(
            [
                f"- Real-trace mean APE {real_trace_repro_mode}: `heuristic={real_trace_summary.get('heuristic', {}).get('mean_ape', 0.0):.3f}`, `score={real_trace_summary.get('score', {}).get('mean_ape', 0.0):.3f}`, `pipeline={real_trace_summary.get('pipeline', {}).get('mean_ape', 0.0):.3f}`.",
                f"- Mode winners on the current bounded replay corpus: {mode_winner_text}.",
                "- The runtime default should only change if one policy wins on both curated scenarios and the bounded real-trace corpus.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- Real-trace replay has not been generated yet. Run `bash scripts/eval_reference_policies.sh` before changing the default policy.",
                "",
            ]
        )
    if mono_stability_summary:
        worst_case = max(
            mono_stability_case_summary.items(),
            key=lambda item: item[1]["std_ape"],
        )
        lines.extend(
            [
                "## Mono Stability Status",
                "",
                f"- Mono repeat-2 mean/std APE: `heuristic={mono_stability_summary.get(('heuristic',), {}).get('mean_ape', 0.0):.3f}±{mono_stability_summary.get(('heuristic',), {}).get('std_ape', 0.0):.3f}`, `score={mono_stability_summary.get(('score',), {}).get('mean_ape', 0.0):.3f}±{mono_stability_summary.get(('score',), {}).get('std_ape', 0.0):.3f}`, `pipeline={mono_stability_summary.get(('pipeline',), {}).get('mean_ape', 0.0):.3f}±{mono_stability_summary.get(('pipeline',), {}).get('std_ape', 0.0):.3f}`.",
                f"- Worst mono variance is `{worst_case[0][0]}/{worst_case[0][1]}` with `std={worst_case[1]['std_ape']:.3f}`. That makes repeat-based comparison mandatory for mono policy changes.",
                "",
            ]
        )
    if mono_repro_summary:
        repro_worst_case = max(
            mono_repro_case_summary.items(),
            key=lambda item: item[1]["std_ape"],
        )
        lines.extend(
            [
                "## Repro Eval Status",
                "",
                f"- Mono repeat-2 with `--repro-eval`: `heuristic={mono_repro_summary.get(('heuristic',), {}).get('mean_ape', 0.0):.3f}±{mono_repro_summary.get(('heuristic',), {}).get('std_ape', 0.0):.3f}`, `score={mono_repro_summary.get(('score',), {}).get('mean_ape', 0.0):.3f}±{mono_repro_summary.get(('score',), {}).get('std_ape', 0.0):.3f}`, `pipeline={mono_repro_summary.get(('pipeline',), {}).get('mean_ape', 0.0):.3f}±{mono_repro_summary.get(('pipeline',), {}).get('std_ape', 0.0):.3f}`.",
                f"- Worst repro mono variance is `{repro_worst_case[0][0]}/{repro_worst_case[0][1]}` with `std={repro_worst_case[1]['std_ape']:.3f}`.",
                "- Repro mode is an evaluation tool, not a runtime default: it exists to separate policy quality from async scheduling noise.",
                "",
            ]
        )
    if full_repro_summary:
        mode_winners = {}
        for mode in sorted({key[0] for key in full_repro_mode_summary}):
            candidates = []
            for policy in POLICIES:
                aggregate = full_repro_mode_summary.get((mode, policy))
                if not aggregate:
                    continue
                candidates.append((aggregate["mean_ape"], aggregate["std_ape"], policy))
            if candidates:
                candidates.sort()
                mode_winners[mode] = candidates[0][2]
        full_repro_worst_case = max(
            full_repro_case_summary.items(),
            key=lambda item: item[1]["std_ape"],
        )
        mode_winner_text = ", ".join(
            f"`{mode}={policy}`" for mode, policy in mode_winners.items()
        )
        lines.extend(
            [
                "## Full Replay Stability Gate",
                "",
                f"- Full repeat-2 with `--repro-eval`: `heuristic={full_repro_summary.get(('heuristic',), {}).get('mean_ape', 0.0):.3f}±{full_repro_summary.get(('heuristic',), {}).get('std_ape', 0.0):.3f}`, `score={full_repro_summary.get(('score',), {}).get('mean_ape', 0.0):.3f}±{full_repro_summary.get(('score',), {}).get('std_ape', 0.0):.3f}`, `pipeline={full_repro_summary.get(('pipeline',), {}).get('mean_ape', 0.0):.3f}±{full_repro_summary.get(('pipeline',), {}).get('std_ape', 0.0):.3f}`.",
                f"- Repeat-2 mode winners on the full bounded corpus: {mode_winner_text}.",
                f"- Worst full-corpus variance is `{full_repro_worst_case[0][0]}/{full_repro_worst_case[0][1]}` with `std={full_repro_worst_case[1]['std_ape']:.3f}`.",
                "- No single policy dominates every mode under repeat replay, so the default stays in `core` and the experiments remain policy candidates instead of migrations.",
                "",
            ]
        )
    if room_focus_summary:
        room_mode_winners = {}
        for mode in sorted({key[0] for key in room_focus_mode_summary}):
            candidates = []
            for policy in POLICIES:
                aggregate = room_focus_mode_summary.get((mode, policy))
                if not aggregate:
                    continue
                candidates.append((aggregate["mean_ape"], aggregate["std_ape"], policy))
            if candidates:
                candidates.sort()
                room_mode_winners[mode] = candidates[0][2]
        room_worst_case = max(
            room_focus_case_summary.items(),
            key=lambda item: item[1]["std_ape"],
        )
        room_mode_winner_text = ", ".join(
            f"`{mode}={policy}`" for mode, policy in room_mode_winners.items()
        )
        lines.extend(
            [
                "## Room Hotspot Status",
                "",
                f"- Room-only repeat-2 with `--repro-eval`: `heuristic={room_focus_summary.get(('heuristic',), {}).get('mean_ape', 0.0):.3f}±{room_focus_summary.get(('heuristic',), {}).get('std_ape', 0.0):.3f}`, `score={room_focus_summary.get(('score',), {}).get('mean_ape', 0.0):.3f}±{room_focus_summary.get(('score',), {}).get('std_ape', 0.0):.3f}`, `pipeline={room_focus_summary.get(('pipeline',), {}).get('mean_ape', 0.0):.3f}±{room_focus_summary.get(('pipeline',), {}).get('std_ape', 0.0):.3f}`.",
                f"- Room-only mode winners: {room_mode_winner_text}.",
                f"- Worst room-only variance is `{room_worst_case[0][0]}/{room_worst_case[0][1]}` with `std={room_worst_case[1]['std_ape']:.3f}`.",
                "- The room hotspot still does not justify a universal migration. It shows that `pipeline` can win locally on room windows while the full-corpus repeat gate still favors `score` overall.",
                "",
            ]
        )
    lines.extend(
        [
        "## Rejected For Now",
        "",
        "- Broad abstract refactors of `tracking` are still off the table. Only decision seams with comparable inputs and metrics graduate into `core`.",
        "- A single canonical implementation is also off the table. `score` and `pipeline` stay intentionally discardable under `src/experiments/`.",
        ]
    )
    DOCS_DIR.joinpath("decisions.md").write_text("\n".join(lines) + "\n")


def generate_interfaces_doc():
    lines = [
        "# Current Minimal Interface",
        "",
        "## Reference Keyframe Policy",
        "",
        "```cpp",
        "struct ReferenceKeyframePolicyInput {",
        "    int tracked_features;",
        "    int detected_keypoints;",
        "    int candidate_landmarks;",
        "    int frames_since_reference;",
        "    int lost_frames;",
        "    bool has_depth;",
        "    bool has_accel;",
        "};",
        "",
        "struct ReferenceKeyframeDecision {",
        "    ReferenceKeyframeAction action;",
        "    double confidence;",
        "    std::string reason;",
        "};",
        "```",
        "",
        "## Why These Fields Survived",
        "",
        "- `tracked_features`: every policy needs current tracking density.",
        "- `detected_keypoints`: separates sparse-image failure from healthy mono frames.",
        "- `candidate_landmarks`: lets policies reason about how much map support the candidate keyframe brings.",
        "- `frames_since_reference`: makes keyframe aging explicit instead of hidden inside each implementation.",
        "- `lost_frames`: keeps recovery pressure observable to the policy instead of coupling it to `Tracking` internals.",
        "- `has_depth`: the primary modality flag that every implementation already relied on.",
        "- `has_accel`: survived once the room follow-up showed `depth` and `depth_accel` windows do not behave the same under replay.",
        "",
        "## Why Other Fields Were Removed",
        "",
        "- Raw `Frame`, `Keyframe`, and `Map` pointers were excluded because they destroy comparability and make policy evaluation depend on global SLAM state.",
        "- Optimizer, loop-closing, and descriptor state were excluded because none of the current variants needed them to make a comparable decision.",
    ]
    DOCS_DIR.joinpath("interfaces.md").write_text("\n".join(lines) + "\n")


def generate_index_doc(rows):
    real_trace_rows = load_real_trace_rows()
    real_trace_summary = aggregate_real_trace_rows(real_trace_rows)
    real_trace_mode_winners = best_real_trace_policy_by_mode(real_trace_rows)
    full_repro_rows = load_full_repro_stability_rows()
    full_repro_summary = aggregate_rows_with_std(full_repro_rows, ["policy"])
    full_repro_mode_summary = aggregate_rows_with_std(full_repro_rows, ["mode", "policy"])
    room_focus_rows = load_room_focus_stability_rows()
    room_focus_summary = aggregate_rows_with_std(room_focus_rows, ["policy"])
    room_focus_mode_summary = aggregate_rows_with_std(room_focus_rows, ["mode", "policy"])

    curated_best = best_accuracy_rows(rows)
    curated_line = ", ".join(
        f"`{row['policy']}` ({float(row['accuracy']):.3f})" for row in curated_best
    )

    full_mode_winners = {}
    for mode in sorted({key[0] for key in full_repro_mode_summary}):
        candidates = []
        for policy in POLICIES:
            aggregate = full_repro_mode_summary.get((mode, policy))
            if not aggregate:
                continue
            candidates.append((aggregate["mean_ape"], aggregate["std_ape"], policy))
        if candidates:
            candidates.sort()
            full_mode_winners[mode] = candidates[0][2]

    room_mode_winners = {}
    for mode in sorted({key[0] for key in room_focus_mode_summary}):
        candidates = []
        for policy in POLICIES:
            aggregate = room_focus_mode_summary.get((mode, policy))
            if not aggregate:
                continue
            candidates.append((aggregate["mean_ape"], aggregate["std_ape"], policy))
        if candidates:
            candidates.sort()
            room_mode_winners[mode] = candidates[0][2]

    lines = [
        "# Reference Keyframe Policy Summary",
        "",
        "This is the GitHub-friendly landing page for the reference-keyframe experiment track.",
        "It summarizes the current conclusion and links to the detailed experiment artifacts.",
        "",
        "## Current Status",
        "",
        f"- Curated corpus top accuracy: {curated_line}.",
        f"- Bounded real-trace replay winner: `score` with mean APE `{real_trace_summary.get('score', {}).get('mean_ape', 0.0):.3f}`.",
        f"- Full repeat-2 replay winner: `score` with mean/std `{full_repro_summary.get(('score',), {}).get('mean_ape', 0.0):.3f}±{full_repro_summary.get(('score',), {}).get('std_ape', 0.0):.3f}`.",
        "- Runtime default is still `heuristic` because no single policy dominates every mode under repeat replay.",
        "",
        "## Mode Snapshot",
        "",
        f"- Single-run bounded replay mode winners: `{', '.join(f'{mode}={policy}' for mode, policy in sorted(real_trace_mode_winners.items()))}`.",
        f"- Repeat-2 bounded replay mode winners: `{', '.join(f'{mode}={policy}' for mode, policy in sorted(full_mode_winners.items()))}`.",
        f"- Room-only hotspot winners: `{', '.join(f'{mode}={policy}' for mode, policy in sorted(room_mode_winners.items()))}`.",
        "",
        "## Read Next",
        "",
        "- [Detailed experiment tables](experiments.md)",
        "- [Current decisions and adoption criteria](decisions.md)",
        "- [Minimal interface that survived the experiments](interfaces.md)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/eval_reference_policies.sh --repeat 1",
        "bash scripts/eval_reference_policies.sh --repeat 2 --output eval_results/reference_keyframe_policy/real_trace_metrics_repeat2.csv",
        "bash scripts/eval_reference_policies.sh --repeat 2 --corpus experiments/reference_keyframe/room_focus_corpus.tsv --output eval_results/reference_keyframe_policy/room_focus_repeat2.csv",
        "./scripts/update_reference_policy_docs.py",
        "```",
    ]
    DOCS_DIR.joinpath("index.md").write_text("\n".join(lines) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    run(["cmake", "--build", str(BUILD_DIR), "-j4", "--target", "reference_policy_experiments"])
    run(
        [
            str(BUILD_DIR / "reference_policy_experiments"),
            str(SCENARIO_FILE),
            str(METRICS_FILE),
            str(DECISIONS_FILE),
        ]
    )
    collect_static_metrics()
    rows = load_joined_results()
    scenarios = read_csv(SCENARIO_FILE)
    generate_experiments_doc(rows, scenarios)
    generate_decisions_doc(rows)
    generate_interfaces_doc()
    generate_index_doc(rows)


if __name__ == "__main__":
    main()
