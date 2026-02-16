# CMP Vectorization Evaluation

## Summary
- status: ok
- num_ok: 28
- num_failed: 2
- avg_rectification_score: 0.5064
- avg_runtime_sec: 0.4359
- avg_window_iou: 0.1396
- avg_door_iou: 0.0198
- avg_window_iou_nonempty_gt: 0.1396
- avg_door_iou_nonempty_gt: 0.0277
- avg_window_count_abs_error: 22.1786
- avg_door_count_abs_error: 3.0000
- door_presence_recall_nonempty_gt: 0.9500
- avg_vector_line_count: 510.0000
- avg_vector_line_manhattan_ratio: 0.9629

## Per Image
| image | status | win_iou | door_iou | win_err | door_err | line_count | line_manhattan | runtime_sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cmp_b0001.jpg | ok | 0.154 | 0.000 | 31 | 5 | 410 | 0.988 | 0.477 |
| cmp_b0014.jpg | ok | 0.150 | 0.000 | 18 | 1 | 593 | 0.978 | 0.615 |
| cmp_b0027.jpg | ok | 0.066 | 0.000 | 18 | 4 | 552 | 0.947 | 0.636 |
| cmp_b0040.jpg | ok | 0.196 | 0.000 | 1 | 8 | 731 | 0.944 | 0.527 |
| cmp_b0053.jpg | ok | 0.241 | 0.000 | 17 | 1 | 628 | 0.960 | 0.578 |
| cmp_b0066.jpg | ok | 0.098 | 0.000 | 247 | 1 | 482 | 0.996 | 0.542 |
| cmp_b0079.jpg | ok | 0.156 | 0.000 | 39 | 1 | 417 | 0.957 | 0.394 |
| cmp_b0092.jpg | ok | 0.140 | 0.100 | 6 | 5 | 435 | 0.979 | 0.423 |
| cmp_b0105.jpg | ok | 0.102 | 0.000 | 25 | 11 | 518 | 0.948 | 0.411 |
| cmp_b0118.jpg | ok | 0.109 | 0.000 | 5 | 0 | 622 | 0.916 | 0.542 |
| cmp_b0131.jpg | ok | 0.102 | 0.000 | 0 | 1 | 682 | 0.985 | 0.608 |
| cmp_b0144.jpg | ok | 0.098 | 0.000 | 10 | 1 | 648 | 0.909 | 0.479 |
| cmp_b0157.jpg | ok | 0.089 | 0.000 | 21 | 0 | 649 | 0.992 | 0.480 |
| cmp_b0170.jpg | ok | 0.148 | 0.000 | 26 | 0 | 945 | 0.979 | 0.545 |
| cmp_b0183.jpg | ok | 0.163 | 0.122 | 8 | 4 | 557 | 0.950 | 0.437 |
| cmp_b0196.jpg | ok | 0.151 | 0.000 | 0 | 0 | 363 | 0.906 | 0.313 |
| cmp_b0209.jpg | ok | 0.135 | 0.000 | 7 | 1 | 456 | 1.000 | 0.433 |
| cmp_b0222.jpg | ok | 0.158 | 0.012 | 12 | 8 | 580 | 0.952 | 0.484 |
| cmp_b0235.jpg | ok | 0.075 | 0.000 | 28 | 2 | 321 | 0.966 | 0.428 |
| cmp_b0248.jpg | ok | 0.143 | 0.000 | 18 | 1 | 645 | 0.955 | 0.438 |
| cmp_b0261.jpg | ok | 0.147 | 0.067 | 17 | 1 | 1139 | 0.977 | 0.629 |
| cmp_b0274.jpg | ok | 0.135 | 0.252 | 28 | 1 | 670 | 0.984 | 0.473 |
| cmp_b0287.jpg | ok | 0.127 | 0.000 | 2 | 6 | 186 | 0.984 | 0.193 |
| cmp_b0300.jpg | ok | 0.120 | 0.000 | 7 | 1 | 239 | 0.941 | 0.216 |
| cmp_b0313.jpg | ok | 0.178 | 0.000 | 10 | 6 | 164 | 0.982 | 0.170 |
| cmp_b0326.jpg | failed | 0.000 | 0.000 | 0 | 0 | 0 | 0.000 | 0.000 |
| cmp_b0339.jpg | ok | 0.139 | 0.000 | 1 | 1 | 313 | 0.923 | 0.384 |
| cmp_b0352.jpg | failed | 0.000 | 0.000 | 0 | 0 | 0 | 0.000 | 0.000 |
| cmp_b0365.jpg | ok | 0.193 | 0.000 | 16 | 3 | 213 | 0.972 | 0.186 |
| cmp_b0378.jpg | ok | 0.194 | 0.000 | 3 | 10 | 122 | 0.992 | 0.164 |
