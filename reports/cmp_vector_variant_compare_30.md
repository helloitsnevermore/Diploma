# Vectorization Variant Comparison (CMP real images)

Selection rule:
- max `avg_openings_iou_nonempty_gt` (average of window/door IoU on nonempty GT)
- then min count errors
- then max `avg_vector_line_manhattan_ratio`
- then min `avg_runtime_sec`

| variant | status | openings_iou_nonempty | win_iou_nonempty | door_iou_nonempty | win_count_err | door_count_err | line_manhattan | line_count | runtime_sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | ok | 0.0833 | 0.1396 | 0.0270 | 22.179 | 3.000 | 0.9471 | 510.00 | 0.5103 |
| strict | ok | 0.0832 | 0.1397 | 0.0267 | 22.179 | 3.000 | 0.9456 | 510.00 | 0.4667 |
| aggressive | ok | 0.0836 | 0.1396 | 0.0277 | 22.179 | 3.000 | 0.9629 | 510.00 | 0.4359 |

Winner: `aggressive`

Source summaries:
- `base`: `reports\cmp_vector_eval_base30_summary.json`
- `strict`: `reports\cmp_vector_eval_strict30_summary.json`
- `aggressive`: `reports\cmp_vector_eval_aggressive30_summary.json`
