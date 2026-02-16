# Segmentation Backend Comparison (CMP real images)

Selection rule:
- max `door_presence_recall_nonempty_gt`
- then max `avg_door_iou_nonempty_gt`
- then max `avg_window_iou_nonempty_gt`
- then min `avg_runtime_sec`
- then min count errors

| backend | status | win_iou_nonempty | door_iou_nonempty | door_presence_recall | win_count_err | door_count_err | runtime_sec |
|---|---|---:|---:|---:|---:|---:|---:|
| contour | ok | 0.1280 | 0.0175 | 0.8333 | 17.000 | 5.300 | 0.5137 |
| yolo_det | ok | 0.0550 | 0.0000 | 0.0000 | 16.400 | 1.200 | 1.1726 |
| yolo_seg | ok | 0.0676 | 0.0000 | 0.0000 | 32.300 | 1.200 | 1.0390 |

Winner: `contour`

Source summaries:
- `contour`: `reports\cmp_real_eval_contour_relaxed10_summary.json`
- `yolo_det`: `reports\cmp_real_eval_yolo_det10_summary.json`
- `yolo_seg`: `reports\cmp_real_eval_yolo_seg10_summary.json`
