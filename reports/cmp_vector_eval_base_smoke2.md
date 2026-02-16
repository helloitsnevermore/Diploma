# CMP Vectorization Evaluation

## Summary
- status: ok
- num_ok: 2
- num_failed: 0
- avg_rectification_score: 0.3583
- avg_runtime_sec: 0.2639
- avg_window_iou: 0.1721
- avg_door_iou: 0.0000
- avg_window_iou_nonempty_gt: 0.1721
- avg_door_iou_nonempty_gt: 0.0000
- avg_window_count_abs_error: 17.0000
- avg_door_count_abs_error: 7.5000
- door_presence_recall_nonempty_gt: 0.0000
- avg_vector_line_count: 266.0000
- avg_vector_line_manhattan_ratio: 0.9833

## Per Image
| image | status | win_iou | door_iou | win_err | door_err | line_count | line_manhattan | runtime_sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cmp_b0001.jpg | ok | 0.156 | 0.000 | 31 | 5 | 410 | 0.983 | 0.387 |
| cmp_b0378.jpg | ok | 0.189 | 0.000 | 3 | 10 | 122 | 0.984 | 0.140 |
