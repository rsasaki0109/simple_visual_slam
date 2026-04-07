# Contributing

We want this project to stay **trustworthy research OSS**: small, reviewable diffs; runnable evaluation hooks; and honest docs. Improvements that tighten reproducibility (tests, `check_regression_gate`, `build_leaderboard`) or clarify behavior usually beat large refactors with no measurement.

## Quick start

1. **Build** (with tests):

   ```bash
   cmake -S . -B build -G Ninja -DBUILD_TESTS=ON
   cmake --build build
   ctest --test-dir build --output-on-failure
   ```

2. **Style**: match existing C++ (`snake_case` files, `PascalCase` classes, English comments where practical). Keep diffs focused on one concern.

3. **Regression checks** (requires local TUM data under `data/tum/` and `evo_ape` on `PATH`):

   ```bash
   python3 scripts/check_regression_gate.py --all-gates --quiet
   ```

   **Comparison baseline (single preset, fast sanity vs external OSS prep):**

   ```bash
   bash scripts/verify_comparison_benchmark.sh xyz_depth
   ```

   This runs every entry in `eval/regression_baselines.json` (several TUM windows; full run is often ~10+ minutes locally). Adjust ceilings in that file only when a change **intentionally** shifts accuracy (document why in the commit message).

## What not to commit

- `data/` (datasets), `build/`, `eval_results/`, large artifacts — see `.gitignore`.

## Maintainers

Release and version bumps: [RELEASING.md](RELEASING.md). User-facing changes: [CHANGELOG.md](CHANGELOG.md).

## License

By contributing, you agree your contributions are under the same **BSD-2-Clause** terms as [LICENSE](LICENSE).
