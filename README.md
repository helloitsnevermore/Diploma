# Diploma: Facade Vectorization MVP

MVP pipeline for `task_1`: video/photos of facade -> rectified frontal orthoimage -> windows/doors + line structure -> regularized vector drawing (`SVG`, optionally `DXF`/`PDF` preview).

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline --input path\to\video.mp4 --outdir runs\run_0001 --config configs\mvp.yaml
```

## CLI

```bash
python -m src.pipeline --input <video_or_images_dir> --outdir <run_dir> [--config configs/mvp.yaml] [--device cpu|cuda] [--seed 42] [--debug]
```

## Output Layout

`--outdir` is always organized as:

- `00_input/frames`, `00_input/manifest.json`
- `10_geometry/facade_ortho.png`, `10_geometry/plane.json`, `10_geometry/quality.json`
- `20_segmentation/instances_windows.json`, `20_segmentation/instances_doors.json`, `20_segmentation/masks_preview.png`
- `30_lines/lines.json`, `30_lines/lines_preview.png`
- `40_vector/result.svg`, `40_vector/result.dxf`, `40_vector/vector_preview.png`, `40_vector/result.pdf`
- `50_report/report.md`, `50_report/timing.json`
- `pipeline.log`

## Notes

- Current implementation is a fully automated baseline with deterministic heuristics and stage contracts.
- It is intentionally structured to replace baseline modules with stronger models/tooling (COLMAP, Detectron2/YOLO, HAWP, optimization-based regularization) without changing CLI or artifacts.

## Real Images Experiment (CMP Facade)

Download and unpack CMP base split:

```powershell
Invoke-WebRequest -Uri "https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip" -OutFile "data/CMP_facade_DB_base.zip"
Expand-Archive -LiteralPath "data/CMP_facade_DB_base.zip" -DestinationPath "data/cmp_facade_base" -Force
```

Run batch evaluation on real facade images:

```bash
python scripts/run_cmp_real_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_real_relaxed30 \
  --report-dir reports \
  --report-prefix cmp_real_eval_relaxed30 \
  --config configs/mvp_real_eval.yaml \
  --limit 30
```

Generated files:

- `reports/cmp_real_eval_relaxed30.csv`
- `reports/cmp_real_eval_relaxed30_summary.json`
- `reports/cmp_real_eval_relaxed30.md`

## Geometry A1 (PyCOLMAP)

Run A1 geometry backend (with automatic fallback to baseline when multi-view reconstruction is not possible):

```bash
python -m src.pipeline --input <video_or_images_dir> --outdir runs/a1_test --config configs/mvp_a1_pycolmap.yaml
```

Compare geometry backends on the same input:

```bash
python scripts/compare_geometry_backends.py \
  --input <video_or_images_dir> \
  --outdir-root runs/geometry_compare \
  --report-dir reports \
  --tag geometry_compare \
  --config configs/mvp.yaml \
  --config configs/mvp_a1_pycolmap.yaml
```
