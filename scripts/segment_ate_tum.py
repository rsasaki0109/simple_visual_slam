#!/usr/bin/env python3
"""Per-segment mean APE (evo_ape) for TUM trajectories.

Splits the *estimated* trajectory into segments (by pose index, equal count per segment),
then runs evo_ape with --t_start/--t_end on each segment (same flags as regression gates).

Usage:
  python3 scripts/segment_ate_tum.py <groundtruth.txt> <trajectory.txt>
  python3 scripts/segment_ate_tum.py <gt> <traj> --segments 5
  python3 scripts/segment_ate_tum.py <gt> <traj> --frame-range 83 166 --segments 4

Options:
  --segments N     Number of equal bins (default: 3).
  --frame-range A B  Use only pose indices [A, B) (0-based, B exclusive). Default: full run.
  --print-full-mean  Also print whole-trajectory mean APE (same evo flags).

Requires: evo on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_lib  # noqa: E402

ROOT = eval_lib.ROOT


def read_tum_ts(path: Path) -> list[float]:
    ts: list[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            ts.append(float(parts[0]))
    return sorted(ts)


def evo_mean_segment(
    gt: Path,
    traj: Path,
    extra: list[str],
    t_start: float,
    t_end: float,
) -> float | None:
    if not shutil.which("evo_ape"):
        sys.stderr.write("evo_ape not found on PATH.\n")
        return None
    tmpdir = Path(tempfile.mkdtemp(prefix="svslam_seg_"))
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
            "--t_start",
            str(t_start),
            "--t_end",
            str(t_end),
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
            return None
        with zipfile.ZipFile(zpath) as zf:
            stats = json.loads(zf.read("stats.json").decode())
        return float(stats["mean"])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("groundtruth", type=Path, help="TUM groundtruth.txt")
    ap.add_argument("trajectory", type=Path, help="Estimated trajectory.txt")
    ap.add_argument("--segments", type=int, default=3, help="Number of equal bins (default 3)")
    ap.add_argument(
        "--frame-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Restrict to pose indices [START, END) (0-based)",
    )
    ap.add_argument(
        "--print-full-mean",
        action="store_true",
        help="Print whole-trajectory mean APE after segments",
    )
    args = ap.parse_args()

    gt = args.groundtruth.resolve()
    traj = args.trajectory.resolve()
    n_seg = args.segments
    if not gt.is_file() or not traj.is_file():
        sys.stderr.write("Missing groundtruth or trajectory.\n")
        sys.exit(1)
    if n_seg < 1:
        sys.stderr.write("--segments must be >= 1\n")
        sys.exit(1)

    ts = read_tum_ts(traj)
    n = len(ts)
    if n < 2:
        sys.stderr.write("Trajectory has too few poses.\n")
        sys.exit(1)

    idx0 = 0
    idx1 = n
    if args.frame_range is not None:
        idx0 = max(0, args.frame_range[0])
        idx1 = min(n, args.frame_range[1])
        if idx1 <= idx0:
            sys.stderr.write("Empty --frame-range after clamping.\n")
            sys.exit(1)

    cfg = json.loads((ROOT / "eval" / "regression_baselines.json").read_text())
    extra = list(cfg.get("evo_ape_extra_args") or [])

    sub_n = idx1 - idx0
    print(f"poses_total={n} analysis=[{idx0}:{idx1}) poses_in_analysis={sub_n} segments={n_seg}")
    print(f"gt={gt}")
    print(f"traj={traj}")
    print(f"evo extras: {' '.join(extra)}")
    print()

    for s in range(n_seg):
        i0 = idx0 + (s * sub_n) // n_seg
        i1 = idx0 + ((s + 1) * sub_n) // n_seg
        if i1 <= i0:
            continue
        t_start = ts[i0]
        t_end = ts[i1 - 1]
        mean = evo_mean_segment(gt, traj, extra, t_start, t_end)
        label = f"frame[{i0}:{i1})"
        if mean is None:
            print(f"bin {s} {label} t=[{t_start:.6f},{t_end:.6f}]  FAIL")
        else:
            print(f"bin {s} {label} t=[{t_start:.6f},{t_end:.6f}]  mean_ape_m={mean:.6f}")

    if args.print_full_mean:
        print()
        full = eval_lib.evo_mean_ape(gt, traj, extra, die_on_error=False)
        if full is not None:
            print(f"full_trajectory mean_ape_m={full:.6f}")
        else:
            print("full_trajectory mean_ape_m= FAIL")


if __name__ == "__main__":
    main()
