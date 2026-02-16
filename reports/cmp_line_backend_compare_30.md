# Line Backend Comparison (CMP real images)

Selection rule:
- max `avg_line_manhattan_ratio`
- then min `avg_runtime_sec`
- then min `avg_line_count`

| backend | status | line_manhattan | line_count | runtime_sec | win_iou_nonempty | door_iou_nonempty |
|---|---|---:|---:|---:|---:|---:|
| lsd | ok | 0.9453 | 510.00 | 0.5666 | 0.1395 | 0.0267 |
| hough | ok | 0.8177 | 1019.07 | 0.5713 | 0.1395 | 0.0267 |

Winner: `lsd`

Source summaries:
- `lsd`: `reports\cmp_real_eval_lines_lsd30_summary.json`
- `hough`: `reports\cmp_real_eval_lines_hough30_summary.json`
