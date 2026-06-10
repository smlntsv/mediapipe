# Parity comparison

`compare.py` reads the web and macOS parity outputs and reports whether they
agree.

## Run

```bash
# After producing both ../shared/output/web.json and macos.json:
python3 compare.py
# custom paths / per-task tolerances / subset of tasks:
python3 compare.py --web ../shared/output/web.json --macos ../shared/output/macos.json \
  --hand-tol 0.02 --face-tol 0.01 --pose-tol 0.04 --tasks hand pose face
```

## What it checks

Enforced (non-zero exit on failure):
- **model SHA256** equality per task (parity is meaningless with different models);
- detection counts (>=1 each, web/macOS instance counts equal);
- per-instance landmark counts: **hand 21, pose 33, face 478**;
- normalized **x/y** within per-task tolerance
  (`--hand-tol` 0.02, `--face-tol` 0.01, `--pose-tol` 0.04), with max/mean abs diff reported;
- hand **handedness** label match.

Report-only (never fail the run):
- the **options** each engine used (diffed and printed);
- **z** differences (normalized depth);
- **worldLandmarks** differences;
- face **blendshapes** / **transformation matrices** counts.

Pose uses a looser default tolerance while full-body / options parity is being
validated. Stdlib only; no dependencies.
