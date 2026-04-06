# Contributing

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

   Adjust ceilings in `eval/regression_baselines.json` only when a change **intentionally** shifts accuracy (document why in the commit message).

## What not to commit

- `data/` (datasets), `build/`, `eval_results/`, large artifacts — see `.gitignore`.

## License

By contributing, you agree your contributions are under the same **BSD-2-Clause** terms as [LICENSE](LICENSE).
