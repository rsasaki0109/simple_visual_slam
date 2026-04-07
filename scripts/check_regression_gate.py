#!/usr/bin/env python3
"""Local regression gate: bitwise trajectory reproducibility + ATE ceiling vs eval/regression_baselines.json.

Requires: built run_mono, optional evo_ape on PATH for ATE check, TUM sequence under data/tum/.
From repo root: python3 scripts/check_regression_gate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_lib  # noqa: E402

ROOT = eval_lib.ROOT
DEFAULT_BASELINE = ROOT / "eval" / "regression_baselines.json"


def run_one_gate(
    cfg: dict,
    gate_name: str,
    *,
    build_dir: Path,
    data_tum: Path,
    skip_ate: bool,
    quiet: bool,
) -> int:
    gates = cfg.get("gates", {})
    if gate_name not in gates:
        sys.stderr.write(f"Unknown gate '{gate_name}' in baseline\n")
        return 1
    gate = gates[gate_name]
    tum_dir = data_tum / gate["tum_sequence"]
    gt = tum_dir / "groundtruth.txt"
    if not tum_dir.is_dir():
        sys.stderr.write(f"Missing dataset: {tum_dir}\n")
        return 1
    if not skip_ate and not gt.is_file():
        sys.stderr.write(f"Missing ground truth: {gt}\n")
        return 1

    sys.stdout.write(f"\n=== gate: {gate_name} ===\n")
    hashes: list[str] = []
    for run in (1, 2):
        eval_lib.clean_traj_artifacts(build_dir)
        sys.stdout.write(f"--- run_mono run {run}/2 ---\n")
        eval_lib.run_slam(build_dir, tum_dir, gate, quiet=quiet)
        traj = build_dir / "trajectory.txt"
        if not traj.is_file() or traj.stat().st_size == 0:
            sys.stderr.write("No trajectory.txt produced.\n")
            return 2
        hashes.append(eval_lib.sha256_file(traj))

    if hashes[0] != hashes[1]:
        sys.stderr.write(
            f"FAIL: trajectory bitwise mismatch between runs\n  {hashes[0]}\n  {hashes[1]}\n"
        )
        return 2

    sys.stdout.write(f"OK: two identical trajectory SHA-256 ({hashes[0][:16]}…)\n")

    if skip_ate:
        return 0

    upper = float(gate["max_mean_ape_m"])
    extra = cfg.get("evo_ape_extra_args") or []
    mean = eval_lib.evo_mean_ape(gt, build_dir / "trajectory.txt", extra, die_on_error=True)
    assert mean is not None
    sys.stdout.write(f"ATE mean (Sim3, policy harness flags): {mean:.6f} m (ceiling {upper:.6f} m)\n")
    if mean > upper:
        sys.stderr.write("FAIL: mean ATE above baseline ceiling (regression).\n")
        return 3
    sys.stdout.write("OK: mean ATE within baseline ceiling.\n")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, default=ROOT / "build", help="CMake build directory (run_mono cwd)")
    ap.add_argument("--data-tum", type=Path, default=ROOT / "data" / "tum", help="Parent of rgbd_dataset_* folders")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument(
        "--gate",
        default="room_mono_head_repro",
        help="Key under gates in baseline JSON (ignored if --all-gates)",
    )
    ap.add_argument(
        "--all-gates",
        action="store_true",
        help="Run every scenario in baseline JSON (sorted by name)",
    )
    ap.add_argument("--skip-ate", action="store_true", help="Only check bitwise reproducibility")
    ap.add_argument("--quiet", action="store_true", help="Suppress run_mono console output")
    args = ap.parse_args()

    cfg = json.loads(args.baseline.read_text())
    gates = cfg.get("gates", {})
    if args.all_gates:
        names = sorted(gates.keys())
    else:
        if args.gate not in gates:
            sys.stderr.write(f"Unknown gate '{args.gate}' in {args.baseline}\n")
            sys.exit(1)
        names = [args.gate]

    build_dir = args.build.resolve()
    for name in names:
        code = run_one_gate(
            cfg,
            name,
            build_dir=build_dir,
            data_tum=args.data_tum,
            skip_ate=args.skip_ate,
            quiet=args.quiet,
        )
        if code != 0:
            sys.exit(code)
    sys.stdout.write("\nAll gates passed.\n")


if __name__ == "__main__":
    main()
