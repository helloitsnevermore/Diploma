# Final MVP Summary (A->E)

- generated_on: 2026-02-15

## Stage Winners
- A (sequential/multi-frame): `pycolmap`
- A (single-image): `homography_baseline`
- B segmentation: `contour`
- C lines: `lsd`
- D vectorization: `aggressive`
- E export: `full`

## A Evidence
- sequential winner metrics: rectification=0.7737, manhattan=0.9000, geometry_sec=32.9728, backend=pycolmap
- single-image winner metrics: rectification=0.5328, manhattan=0.4950, geometry_sec=0.3711, backend=homography_baseline

## Final Configs
- default: `configs/mvp_final.yaml`
- quality: `configs/mvp_final_quality.yaml`

## One-Command Run
```bash
python scripts/run_mvp_final.py --input <video_or_images_dir> --outdir runs/run_final_0001 --profile default --device cpu --seed 42
python scripts/run_mvp_final.py --input <video_or_images_dir> --outdir runs/run_final_quality_0001 --profile quality --device cpu --seed 42
```

## Notes
- Stage A sequential comparison: PyCOLMAP has higher rectification score but much slower runtime.
- Stage A single-image comparison: PyCOLMAP profile falls back to homography baseline.
- CMP 30-image evaluations across B-E have 2 geometry-gate failures (cmp_b0326, cmp_b0352).
- Default final profile prioritizes robustness/speed on mixed single-image and short-sequence inputs.

## Source Reports
- `reports/geometry_compare_seq.json`
- `reports/geometry_compare_single.json`
- `reports/cmp_seg_backend_compare_30.json`
- `reports/cmp_line_backend_compare_30.json`
- `reports/cmp_vector_variant_compare_30.json`
- `reports/cmp_export_variant_compare_30.json`
