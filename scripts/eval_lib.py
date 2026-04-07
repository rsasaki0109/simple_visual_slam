"""Shared helpers for SLAM evaluation scripts (run_mono + evo_ape)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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

    cmd = [str(exe), "--tum", str(tum_dir)]
    if gate.get("use_depth"):
        cmd.append("--depth")
    if gate.get("use_accel"):
        cmd.append("--accel")
    cmd.extend(
        [
            "--reference-policy",
            str(gate["reference_policy"]),
            "--skip-frames",
            str(int(gate["skip_frames"])),
            "--max-frames",
            str(int(gate["max_frames"])),
            "--no-viz",
        ]
    )
    if gate.get("repro_eval", True):
        cmd.append("--repro-eval")

    kw: dict = {"cwd": build_dir, "check": True}
    if quiet:
        kw["stdout"] = subprocess.DEVNULL
        kw["stderr"] = subprocess.DEVNULL
        kw["stdin"] = subprocess.DEVNULL
    subprocess.run(cmd, **kw)


def evo_mean_ape(gt: Path, traj: Path, extra: list[str], *, die_on_error: bool = True) -> float | None:
    if not shutil.which("evo_ape"):
        sys.stderr.write("evo_ape not found on PATH; install evo (pip install evo).\n")
        if die_on_error:
            sys.exit(1)
        return None
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
            if die_on_error:
                sys.exit(1)
            return None
        with zipfile.ZipFile(zpath) as zf:
            stats = json.loads(zf.read("stats.json").decode())
        return float(stats["mean"])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
