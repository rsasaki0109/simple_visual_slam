#!/usr/bin/env python3
"""Run a lightweight SLAM smoke test against the CI dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_lib  # noqa: E402


def count_tum_poses(path: Path) -> int:
    count = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        count += 1
    return count


def load_xyz_by_timestamp(path: Path) -> dict[float, tuple[float, float, float]]:
    poses: dict[float, tuple[float, float, float]] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        poses[float(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return poses


def mean_ate_sim3(gt_path: Path, traj_path: Path) -> float | None:
    try:
        import numpy as np
    except Exception:
        return None

    gt_by_ts = load_xyz_by_timestamp(gt_path)
    traj_by_ts = load_xyz_by_timestamp(traj_path)
    common_ts = sorted(set(gt_by_ts).intersection(traj_by_ts))
    if len(common_ts) < 3:
        raise RuntimeError("Need at least 3 aligned poses for Sim3 ATE")

    gt = np.asarray([gt_by_ts[ts] for ts in common_ts], dtype=float)
    est = np.asarray([traj_by_ts[ts] for ts in common_ts], dtype=float)

    gt_mean = gt.mean(axis=0)
    est_mean = est.mean(axis=0)
    gt_centered = gt - gt_mean
    est_centered = est - est_mean

    covariance = (gt_centered.T @ est_centered) / len(common_ts)
    u, singular_values, vt = np.linalg.svd(covariance)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        s[-1, -1] = -1.0

    rotation = u @ s @ vt
    variance = np.mean(np.sum(est_centered * est_centered, axis=1))
    if variance <= 0.0:
        raise RuntimeError("Estimated trajectory variance is zero")
    scale = np.trace(np.diag(singular_values) @ s) / variance
    translation = gt_mean - scale * (rotation @ est_mean)

    aligned = scale * (est @ rotation.T) + translation
    errors = np.linalg.norm(aligned - gt, axis=1)
    return float(errors.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, default=ROOT / "build", help="CMake build directory")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ci_test", help="CI test dataset directory")
    ap.add_argument("--expected-poses", type=int, default=10, help="Expected trajectory pose count")
    ap.add_argument("--max-frames", type=int, default=10, help="Frame budget for run_mono")
    ap.add_argument(
        "--max-mean-ate-m",
        type=float,
        default=None,
        help="Optional loose Sim3-aligned ATE ceiling in meters",
    )
    args = ap.parse_args()

    build_dir = args.build.resolve()
    data_dir = args.data.resolve()
    exe = build_dir / "run_mono"
    if not exe.is_file():
        raise SystemExit(f"Missing {exe}; build the project first.")
    if not data_dir.is_dir():
        raise SystemExit(f"Missing dataset directory: {data_dir}")

    eval_lib.clean_traj_artifacts(build_dir)

    cmd = [
        str(exe),
        "--tum",
        str(data_dir),
        "--tum-camera-config",
        str(data_dir / "camera.json"),
        "--max-frames",
        str(args.max_frames),
        "--repro-eval",
        "--no-viz",
    ]
    subprocess.run(cmd, cwd=build_dir, check=True)

    trajectory_path = build_dir / "trajectory.txt"
    if not trajectory_path.is_file() or trajectory_path.stat().st_size == 0:
        raise SystemExit("Smoke test failed: trajectory.txt was not produced.")

    pose_count = count_tum_poses(trajectory_path)
    if pose_count != args.expected_poses:
        raise SystemExit(
            f"Smoke test failed: expected {args.expected_poses} poses in trajectory.txt, got {pose_count}."
        )

    print(f"Smoke test OK: trajectory.txt contains {pose_count} poses.")

    if args.max_mean_ate_m is not None:
        gt_path = data_dir / "groundtruth.txt"
        mean_ate = mean_ate_sim3(gt_path, trajectory_path)
        if mean_ate is None:
            print("ATE check skipped: numpy not available.")
            return
        print(f"Sim3 mean ATE: {mean_ate:.6f} m")
        if mean_ate > args.max_mean_ate_m:
            raise SystemExit(
                f"Smoke test failed: mean ATE {mean_ate:.6f} m exceeds ceiling {args.max_mean_ate_m:.6f} m."
            )


if __name__ == "__main__":
    main()
