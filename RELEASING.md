# Releasing

Version numbers follow **CMake `project(SimpleVisualSLAM VERSION x.y.z)`** and are shown by:

```bash
./build/run_mono --version
```

## Checklist for a tagged release

1. Bump `VERSION` in the root `CMakeLists.txt` (`project(... VERSION x.y.z)`).
2. Match **`CITATION.cff`**: set `version` and `date-released` to the same release (GitHub’s “Cite this repository” uses this file).
3. Update `CHANGELOG.md`: move items from `[Unreleased]` under a new `[x.y.z]` section with date.
4. Reconfigure and rebuild; run `ctest --test-dir build --output-on-failure`.
5. With TUM data and `evo_ape` available, run `python3 scripts/check_regression_gate.py --all-gates --quiet`. Adjust `eval/regression_baselines.json` ceilings only when the change intentionally moves benchmarks (document in the commit).
6. Tag: `git tag -a vx.y.z -m "Release vx.y.z"` and push tags.
7. Publish release notes on GitHub (can paste from `CHANGELOG.md`).

Pre-1.0 releases (0.y.z) may include breaking changes; document them in the changelog.
