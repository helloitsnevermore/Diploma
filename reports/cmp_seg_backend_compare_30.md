# Segmentation Backend Comparison (CMP real images)

Selection rule:
- max `door_presence_recall_nonempty_gt`
- then max `avg_door_iou_nonempty_gt`
- then max `avg_window_iou_nonempty_gt`
- then min `avg_runtime_sec`
- then min count errors

| backend | status | win_iou_nonempty | door_iou_nonempty | door_presence_recall | win_count_err | door_count_err | runtime_sec |
|---|---|---:|---:|---:|---:|---:|---:|
| contour | ok | 0.1395 | 0.0267 | 0.9500 | 22.179 | 3.000 | 0.5830 |
| yolo_det | ok | 0.0356 | 0.0000 | 0.0000 | 29.821 | 1.500 | 0.7878 |
| yolo_seg | ok | 0.0587 | 0.0000 | 0.0000 | 32.179 | 1.500 | 1.0559 |

Winner: `contour`

Source summaries:
- `contour`: `reports\cmp_real_eval_contour_relaxed30_v2_summary.json`
- `yolo_det`: `reports\cmp_real_eval_yolo_det30_summary.json`
- `yolo_seg`: `reports\cmp_real_eval_yolo_seg30_summary.json`
