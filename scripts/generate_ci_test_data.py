#!/usr/bin/env python3
"""Generate a deterministic tiny TUM-style dataset for CI smoke tests."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "ci_test"

WIDTH = 640
HEIGHT = 480
FX = 517.3
FY = 516.5
CX = 318.6
CY = 255.3


def matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def transpose3(m: list[list[float]]) -> list[list[float]]:
    return [[m[c][r] for c in range(3)] for r in range(3)]


def rotation_matrix_to_quaternion(r: list[list[float]]) -> tuple[float, float, float, float]:
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
        return qx, qy, qz, qw
    if r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
        return qx, qy, qz, qw
    if r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
        return qx, qy, qz, qw
    s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
    qw = (r[1][0] - r[0][1]) / s
    qx = (r[0][2] + r[2][0]) / s
    qy = (r[1][2] + r[2][1]) / s
    qz = 0.25 * s
    return qx, qy, qz, qw


def write_pgm(path: Path, image: list[int]) -> None:
    with path.open("wb") as f:
        f.write(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode())
        f.write(bytes(image))


def make_marker_pattern(point_id: int) -> list[list[int]]:
    rng = random.Random(point_id * 7919 + 17)
    pattern: list[list[int]] = []
    for y in range(9):
        row = []
        for x in range(9):
            if x in (0, 8) or y in (0, 8):
                row.append(20)
            elif x in (1, 7) or y in (1, 7):
                row.append(235)
            else:
                row.append(40 if rng.random() < 0.5 else 210)
        pattern.append(row)
    return pattern


def build_scene(point_count: int, seed: int) -> list[tuple[float, float, float, int]]:
    rng = random.Random(seed)
    points: list[tuple[float, float, float, int]] = []
    for point_id in range(point_count):
        points.append(
            (
                rng.uniform(-1.8, 1.8),
                rng.uniform(-1.3, 1.3),
                rng.uniform(3.0, 6.5),
                point_id,
            )
        )
    return points


def render_frame(
    scene: list[tuple[float, float, float, int]],
    patterns: dict[int, list[list[int]]],
    frame_idx: int,
) -> tuple[list[int], tuple[float, float, float], tuple[float, float, float, float]]:
    tx = 0.035 * frame_idx
    ty = 0.0
    tz = 0.0
    yaw = 0.010 * frame_idx
    pitch = 0.002 * math.sin(frame_idx * 0.4)
    roll = 0.0015 * math.cos(frame_idx * 0.3)

    cyaw = math.cos(yaw)
    syaw = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    rz = [[cyaw, -syaw, 0.0], [syaw, cyaw, 0.0], [0.0, 0.0, 1.0]]
    ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]

    r_wc = matmul3(matmul3(rz, ry), rx)
    r_cw = transpose3(r_wc)
    qx, qy, qz, qw = rotation_matrix_to_quaternion(r_wc)

    image = [0] * (WIDTH * HEIGHT)
    for y in range(HEIGHT):
        base = 90 + (y * 30) // HEIGHT
        row_offset = y * WIDTH
        for x in range(WIDTH):
            image[row_offset + x] = min(255, max(0, base + ((x * 20) // WIDTH)))

    for x_w, y_w, z_w, point_id in scene:
        dx = x_w - tx
        dy = y_w - ty
        dz = z_w - tz
        x_c = r_cw[0][0] * dx + r_cw[0][1] * dy + r_cw[0][2] * dz
        y_c = r_cw[1][0] * dx + r_cw[1][1] * dy + r_cw[1][2] * dz
        z_c = r_cw[2][0] * dx + r_cw[2][1] * dy + r_cw[2][2] * dz
        if z_c <= 0.5:
            continue

        u = FX * (x_c / z_c) + CX
        v = FY * (y_c / z_c) + CY
        u_i = int(round(u))
        v_i = int(round(v))
        if u_i < 6 or u_i >= WIDTH - 6 or v_i < 6 or v_i >= HEIGHT - 6:
            continue

        pattern = patterns[point_id]
        scale = 2 if z_c < 3.6 else 1
        half = 4 * scale
        for py in range(-half, half + 1):
            src_y = (py + half) * 8 // (2 * half + 1)
            yy = v_i + py
            row_offset = yy * WIDTH
            pattern_row = pattern[src_y]
            for px in range(-half, half + 1):
                src_x = (px + half) * 8 // (2 * half + 1)
                xx = u_i + px
                value = pattern_row[src_x]
                idx = row_offset + xx
                current = image[idx]
                if abs(value - 128) > abs(current - 128):
                    image[idx] = value

    return image, (tx, ty, tz), (qx, qy, qz, qw)


def generate_dataset(output_dir: Path, frames: int, point_count: int, seed: int) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "rgb").mkdir(parents=True)

    camera = {
        "fx": FX,
        "fy": FY,
        "cx": CX,
        "cy": CY,
        "width": WIDTH,
        "height": HEIGHT,
        "distortion": [],
    }
    (output_dir / "camera.json").write_text(json.dumps(camera, indent=2) + "\n")

    manifest = {
        "frames": frames,
        "expected_poses": frames,
        "seed": seed,
        "point_count": point_count,
        "camera_config": "camera.json",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    scene = build_scene(point_count=point_count, seed=seed)
    patterns = {point_id: make_marker_pattern(point_id) for _, _, _, point_id in scene}

    rgb_lines = ["# synthetic ci smoke dataset\n"]
    gt_lines = ["# timestamp tx ty tz qx qy qz qw\n"]

    for frame_idx in range(frames):
        timestamp = 1.0 + frame_idx * (1.0 / 30.0)
        image, translation, quat = render_frame(scene, patterns, frame_idx)
        name = f"{timestamp:.6f}.pgm"
        rel_path = f"rgb/{name}"
        write_pgm(output_dir / rel_path, image)
        rgb_lines.append(f"{timestamp:.6f} {rel_path}\n")
        gt_lines.append(
            f"{timestamp:.6f} {translation[0]:.9f} {translation[1]:.9f} {translation[2]:.9f} "
            f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}\n"
        )

    (output_dir / "rgb.txt").write_text("".join(rgb_lines))
    (output_dir / "groundtruth.txt").write_text("".join(gt_lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output dataset directory")
    ap.add_argument("--frames", type=int, default=10, help="Number of RGB frames to generate")
    ap.add_argument("--point-count", type=int, default=900, help="Number of synthetic 3D markers")
    ap.add_argument("--seed", type=int, default=12345, help="Scene seed")
    args = ap.parse_args()

    output_dir = args.output.resolve()
    generate_dataset(output_dir, frames=args.frames, point_count=args.point_count, seed=args.seed)
    print(f"Generated CI smoke dataset at {output_dir}")


if __name__ == "__main__":
    main()
