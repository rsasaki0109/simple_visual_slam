#!/usr/bin/env python3
"""Local regression gate: bitwise trajectory reproducibility + ATE ceiling vs eval/regression_baselines.json.

Requires: built run_mono, optional evo_ape on PATH for ATE check, TUM sequence under data/tum/.
From repo root: python3 scripts/check_regression_gate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "eval" / "regression_baselines.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_traj_artifacts(build_dir: Path) -> None:
    for name in ("trajectory.txt", "trajectory_online.txt", "trajectory_keyframes.txt", "map.bin"):
        p = build_dir / name
        if p.exists():
            p.unlink()


def run_slam(build_dir: Path, tum_dir: Path, gate: dict, *, quiet: bool) -> None:
    exe = build_dir / "run_mono"
    if not exe.is_file():
        sys.stderr.write(f"Missing {exe}; build the project first.\n")
        sys.exit(1)

    cmd = [
        str(exe),
        "--tum",
        str(tum_dir),
        "--reference-policy",
        str(gate["reference_policy"]),
        "--skip-frames",
        str(int(gate["skip_frames"])),
        "--max-frames",
        str(int(gate["max_frames"])),
        "--no-viz",
    ]
    if gate.get("repro_eval", True):
        cmd.append("--repro-eval")

    kw = {"cwd": build_dir, "check": True}
    if quiet:
        kw["stdout"] = subprocess.DEVNULL
        kw["stderr"] = subprocess.DEVNULL
        kw["stdin"] = subprocess.DEVNULL
    subprocess.run(cmd, **kw)


def evo_mean_ape(gt: Path, traj: Path, extra: list[str]) -> float:
    if not shutil.which("evo_ape"):
        sys.stderr.write("evo_ape not found on PATH; install evo or use --skip-ate.\n")
        sys.exit(1)
    tmpdir = Path(tempfile.mkdtemp(prefix="svslam_evo_"))
    zpath = tmpdir / "results.zip"
    try:
        cmd = [
            "evo_ape",
            "tum",
            str(gt),
            str(traj),
            "--save_results",
            str(zpath),
            "--silent",
            *extra,
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr or proc.stdout or "evo_ape failed\n")
            sys.exit(1)
        with zipfile.ZipFile(zpath) as zf:
            stats = json.loads(zf.read("stats.json").decode())
        return float(stats["mean"])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", type=Path, default=ROOT / "build", help="CMake build directory (run_mono cwd)")
    ap.add_argument("--data-tum", type=Path, default=ROOT / "data" / "tum", help="Parent of rgbd_dataset_* folders")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--gate", default="room_mono_head_repro", help="Key under gates in baseline JSON")
    ap.add_argument("--skip-ate", action="store_true", help="Only check bitwise reproducibility")
    ap.add_argument("--quiet", action="store_true", help="Suppress run_mono console output")
    args = ap.parse_args()

    cfg = json.loads(args.baseline.read_text())
    if args.gate not in cfg.get("gates", {}):
        sys.stderr.write(f"Unknown gate '{args.gate}' in {args.baseline}\n")
        sys.exit(1)
    gate = cfg["gates"][args.gate]
    tum_name = gate["tum_sequence"]
    tum_dir = args.data_tum / tum_name
    gt = tum_dir / "groundtruth.txt"
    if not tum_dir.is_dir():
        sys.stderr.write(f"Missing dataset: {tum_dir}\n")
        sys.exit(1)
    if not args.skip_ate and not gt.is_file():
        sys.stderr.write(f"Missing ground truth: {gt}\n")
        sys.exit(1)

    build_dir = args.build.resolve()
    hashes = []
    for run in (1, 2):
        clean_traj_artifacts(build_dir)
        sys.stdout.write(f"--- run_mono run {run}/2 ---\n")
        run_slam(build_dir, tum_dir, gate, quiet=args.quiet)
        traj = build_dir / "trajectory.txt"
        if not traj.is_file() or traj.stat().st_size == 0:
            sys.stderr.write("No trajectory.txt produced.\n")
            sys.exit(2)
        hashes.append(sha256_file(traj))

    if hashes[0] != hashes[1]:
        sys.stderr.write(
            f"FAIL: trajectory bitwise mismatch between runs\n  {hashes[0]}\n  {hashes[1]}\n"
        )
        sys.exit(2)

    sys.stdout.write(f"OK: two identical trajectory SHA-256 ({hashes[0][:16]}…)\n")

    if args.skip_ate:
        return

    upper = float(gate["max_mean_ape_m"])
    extra = cfg.get("evo_ape_extra_args") or []
    mean = evo_mean_ape(gt, build_dir / "trajectory.txt", extra)
    sys.stdout.write(f"ATE mean (Sim3, policy harness flags): {mean:.6f} m (ceiling {upper:.6f} m)\n")
    if mean > upper:
        sys.stderr.write("FAIL: mean ATE above baseline ceiling (regression).\n")
        sys.exit(3)
    sys.stdout.write("OK: mean ATE within baseline ceiling.\n")


if __name__ == "__main__":
    main()
