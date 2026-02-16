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

## Stage B Backend Comparison (Contour vs YOLO)

After producing per-backend summary files, build a single comparison report.
Quick check (10 images):

```bash
python scripts/compare_segmentation_backends.py \
  --entry contour=reports/cmp_real_eval_contour_relaxed10_summary.json \
  --entry yolo_det=reports/cmp_real_eval_yolo_det10_summary.json \
  --entry yolo_seg=reports/cmp_real_eval_yolo_seg10_summary.json \
  --report-dir reports \
  --tag cmp_seg_backend_compare
```

Representative check (30 images):

```bash
python scripts/compare_segmentation_backends.py \
  --entry contour=reports/cmp_real_eval_contour_relaxed30_v2_summary.json \
  --entry yolo_det=reports/cmp_real_eval_yolo_det30_summary.json \
  --entry yolo_seg=reports/cmp_real_eval_yolo_seg30_summary.json \
  --report-dir reports \
  --tag cmp_seg_backend_compare_30
```

Generated files:

- `reports/cmp_seg_backend_compare.json`
- `reports/cmp_seg_backend_compare.md`
- `reports/cmp_seg_backend_compare_30.json`
- `reports/cmp_seg_backend_compare_30.md`

Current result on CMP (30 images, evaluated on 2026-02-15): winner is `contour`.

## Stage C Backend Comparison (LSD vs Hough)

Run same real-image evaluation with line backend switched:

```bash
python scripts/run_cmp_real_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_real_lines_lsd30 \
  --report-dir reports \
  --report-prefix cmp_real_eval_lines_lsd30 \
  --config configs/mvp_real_eval.yaml \
  --limit 30

python scripts/run_cmp_real_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_real_lines_hough30 \
  --report-dir reports \
  --report-prefix cmp_real_eval_lines_hough30 \
  --config configs/mvp_c_hough.yaml \
  --limit 30
```

Build line-backend comparison report:

```bash
python scripts/compare_line_backends.py \
  --entry lsd=reports/cmp_real_eval_lines_lsd30_summary.json \
  --entry hough=reports/cmp_real_eval_lines_hough30_summary.json \
  --report-dir reports \
  --tag cmp_line_backend_compare_30
```

Generated files:

- `reports/cmp_line_backend_compare_30.json`
- `reports/cmp_line_backend_compare_30.md`

Current result on CMP (30 images, evaluated on 2026-02-15): winner is `lsd`.

## Stage D Vectorization Comparison (Regularization/Snapping)

Run vectorization-focused evaluation (metrics are computed from `40_vector/result.svg`):

```bash
python scripts/run_cmp_vector_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_vector_base30 \
  --report-dir reports \
  --report-prefix cmp_vector_eval_base30 \
  --config configs/mvp_d_vector_base.yaml \
  --limit 30

python scripts/run_cmp_vector_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_vector_strict30 \
  --report-dir reports \
  --report-prefix cmp_vector_eval_strict30 \
  --config configs/mvp_d_vector_strict.yaml \
  --limit 30

python scripts/run_cmp_vector_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_vector_aggressive30 \
  --report-dir reports \
  --report-prefix cmp_vector_eval_aggressive30 \
  --config configs/mvp_d_vector_aggressive.yaml \
  --limit 30
```

Build variant comparison report:

```bash
python scripts/compare_vectorization_variants.py \
  --entry base=reports/cmp_vector_eval_base30_summary.json \
  --entry strict=reports/cmp_vector_eval_strict30_summary.json \
  --entry aggressive=reports/cmp_vector_eval_aggressive30_summary.json \
  --report-dir reports \
  --tag cmp_vector_variant_compare_30
```

Generated files:

- `reports/cmp_vector_eval_base30_summary.json`
- `reports/cmp_vector_eval_strict30_summary.json`
- `reports/cmp_vector_eval_aggressive30_summary.json`
- `reports/cmp_vector_variant_compare_30.json`
- `reports/cmp_vector_variant_compare_30.md`

Current result on CMP (30 images, evaluated on 2026-02-15): winner is `aggressive`.
Default pipeline configs (`configs/mvp.yaml`, `configs/mvp_real_eval.yaml`) were updated to this vectorization setting.

## Stage E Export Comparison (SVG/DXF/PDF Packaging)

Run export validation on real images for three export profiles:

```bash
python scripts/run_cmp_export_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_export_full30 \
  --report-dir reports \
  --report-prefix cmp_export_eval_full30 \
  --config configs/mvp_e_full.yaml \
  --limit 30

python scripts/run_cmp_export_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_export_no_pdf30 \
  --report-dir reports \
  --report-prefix cmp_export_eval_no_pdf30 \
  --config configs/mvp_e_no_pdf.yaml \
  --limit 30

python scripts/run_cmp_export_eval.py \
  --dataset-dir data/cmp_facade_base/base \
  --outdir-root runs/cmp_export_svg_only30 \
  --report-dir reports \
  --report-prefix cmp_export_eval_svg_only30 \
  --config configs/mvp_e_svg_only.yaml \
  --limit 30
```

Build export-profile comparison report:

```bash
python scripts/compare_export_variants.py \
  --entry full=reports/cmp_export_eval_full30_summary.json \
  --entry no_pdf=reports/cmp_export_eval_no_pdf30_summary.json \
  --entry svg_only=reports/cmp_export_eval_svg_only30_summary.json \
  --report-dir reports \
  --tag cmp_export_variant_compare_30
```

Generated files:

- `reports/cmp_export_eval_full30_summary.json`
- `reports/cmp_export_eval_no_pdf30_summary.json`
- `reports/cmp_export_eval_svg_only30_summary.json`
- `reports/cmp_export_variant_compare_30.json`
- `reports/cmp_export_variant_compare_30.md`

Current result on CMP (30 images, evaluated on 2026-02-15): winner is `full`.
Selection rule: maximize outputs completeness (`SVG+DXF+PDF`) first, then validation rate and runtime.

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

## Finalization (A->E Locked Profiles)

Build consolidated final report from all stage comparison artifacts:

```bash
python scripts/build_final_mvp_report.py --report-dir reports --tag mvp_final_summary
```

Run final one-command pipeline profile:

```bash
python scripts/run_mvp_final.py --input <video_or_images_dir> --outdir runs/run_final_0001 --profile default --device cpu --seed 42
python scripts/run_mvp_final.py --input <video_or_images_dir> --outdir runs/run_final_quality_0001 --profile quality --device cpu --seed 42
```

Generated files:

- `reports/mvp_final_summary.json`
- `reports/mvp_final_summary.md`
- `configs/mvp_final.yaml`
- `configs/mvp_final_quality.yaml`

Final winners on CMP (30 images, consolidated on 2026-02-15):

- A sequential: `pycolmap`
- A single-image: `homography_baseline`
- B segmentation: `contour`
- C lines: `lsd`
- D vectorization: `aggressive`
- E export: `full`
