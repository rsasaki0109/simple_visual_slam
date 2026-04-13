#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# CI currently uses a deterministic synthetic fallback instead of fetching a remote
# TUM archive so the smoke test stays fast and avoids external availability issues.
python3 "$ROOT_DIR/scripts/generate_ci_test_data.py" --output "$ROOT_DIR/data/ci_test"
