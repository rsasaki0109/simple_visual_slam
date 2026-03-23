#!/usr/bin/env python3

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w, 2.0 * x * z + 2.0 * y * w],
        [2.0 * x * y + 2.0 * z * w, 1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w],
        [2.0 * x * z - 2.0 * y * w, 2.0 * y * z + 2.0 * x * w, 1.0 - 2.0 * x * x - 2.0 * y * y],
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=Path("map.bin"))
    parser.add_argument("--output", type=Path, default=Path("trajectory_keyframes.txt"))
    args = parser.parse_args()

    kfs = []
    with args.map.open("rb") as f:
        if f.read(6) != b"SVSLAM":
            raise RuntimeError("invalid map header")

        f.read(9 * 8)  # camera block
        num_kfs = struct.unpack("Q", f.read(8))[0]

        ulong_size = struct.calcsize("L")
        for _ in range(num_kfs):
            kf_id = struct.unpack("L", f.read(ulong_size))[0]
            timestamp = struct.unpack("d", f.read(8))[0]
            t = np.frombuffer(f.read(8 * 3), dtype=np.float64).copy()
            q = np.frombuffer(f.read(8 * 4), dtype=np.float64).copy()  # x y z w

            rotation = quaternion_to_rotation_matrix(q[0], q[1], q[2], q[3])
            camera_center = -rotation.T @ t
            q_wc = np.array([-q[0], -q[1], -q[2], q[3]])

            num_kps = struct.unpack("Q", f.read(8))[0]
            f.read(num_kps * (4 + 4 + 4 + 4))
            rows, cols, _ = struct.unpack("iii", f.read(12))
            if rows > 0 and cols > 0:
                f.read(rows * cols)
            f.read(num_kps * 8)

            kfs.append((timestamp, kf_id, camera_center, q_wc))

    kfs.sort(key=lambda item: (item[0], item[1]))

    with args.output.open("w", encoding="utf-8") as f:
        f.write("# timestamp x y z qx qy qz qw\n")
        for timestamp, _, pos, quat in kfs:
            f.write(
                f"{timestamp:.6f} "
                f"{pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}\n"
            )


if __name__ == "__main__":
    main()
