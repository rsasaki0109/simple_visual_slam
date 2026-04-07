#!/usr/bin/env python3
"""Print mean ATE (m) with the same evo_ape flags as eval/regression_baselines.json.

Usage:
  python3 scripts/print_ate_mean.py <groundtruth_tum.txt> <trajectory_tum.txt>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import eval_lib  # noqa: E402

ROOT = eval_lib.ROOT


def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write(__doc__ or "")
        sys.exit(2)
    gt = Path(sys.argv[1]).resolve()
    traj = Path(sys.argv[2]).resolve()
    if not gt.is_file() or not traj.is_file():
        sys.stderr.write("Missing groundtruth or trajectory file.\n")
        sys.exit(1)
    cfg = json.loads((ROOT / "eval" / "regression_baselines.json").read_text())
    extra = cfg.get("evo_ape_extra_args") or []
    mean = eval_lib.evo_mean_ape(gt, traj, extra, die_on_error=True)
    assert mean is not None
    print(f"{mean:.8f}")


if __name__ == "__main__":
    main()
