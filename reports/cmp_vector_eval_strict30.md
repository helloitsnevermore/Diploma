# CMP Vectorization Evaluation

## Summary
- status: ok
- num_ok: 28
- num_failed: 2
- avg_rectification_score: 0.5064
- avg_runtime_sec: 0.4667
- avg_window_iou: 0.1397
- avg_door_iou: 0.0191
- avg_window_iou_nonempty_gt: 0.1397
- avg_door_iou_nonempty_gt: 0.0267
- avg_window_count_abs_error: 22.1786
- avg_door_count_abs_error: 3.0000
- door_presence_recall_nonempty_gt: 0.9500
- avg_vector_line_count: 510.0000
- avg_vector_line_manhattan_ratio: 0.9456

## Per Image
| image | status | win_iou | door_iou | win_err | door_err | line_count | line_manhattan | runtime_sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cmp_b0001.jpg | ok | 0.155 | 0.000 | 31 | 5 | 410 | 0.983 | 0.408 |
| cmp_b0014.jpg | ok | 0.150 | 0.000 | 18 | 1 | 593 | 0.954 | 0.565 |
| cmp_b0027.jpg | ok | 0.068 | 0.000 | 18 | 4 | 552 | 0.929 | 0.484 |
| cmp_b0040.jpg | ok | 0.196 | 0.000 | 1 | 8 | 731 | 0.911 | 0.567 |
| cmp_b0053.jpg | ok | 0.241 | 0.005 | 17 | 1 | 628 | 0.952 | 1.118 |
| cmp_b0066.jpg | ok | 0.102 | 0.000 | 247 | 1 | 482 | 0.985 | 0.702 |
| cmp_b0079.jpg | ok | 0.154 | 0.000 | 39 | 1 | 417 | 0.887 | 0.499 |
| cmp_b0092.jpg | ok | 0.141 | 0.098 | 6 | 5 | 435 | 0.968 | 0.406 |
| cmp_b0105.jpg | ok | 0.101 | 0.000 | 25 | 11 | 518 | 0.923 | 0.391 |
| cmp_b0118.jpg | ok | 0.109 | 0.000 | 5 | 0 | 622 | 0.902 | 0.455 |
| cmp_b0131.jpg | ok | 0.100 | 0.000 | 0 | 1 | 682 | 0.963 | 0.484 |
| cmp_b0144.jpg | ok | 0.098 | 0.000 | 10 | 1 | 648 | 0.898 | 0.482 |
| cmp_b0157.jpg | ok | 0.091 | 0.000 | 21 | 0 | 649 | 0.985 | 0.458 |
| cmp_b0170.jpg | ok | 0.148 | 0.000 | 26 | 0 | 945 | 0.975 | 0.590 |
| cmp_b0183.jpg | ok | 0.164 | 0.121 | 8 | 4 | 557 | 0.937 | 0.469 |
| cmp_b0196.jpg | ok | 0.152 | 0.000 | 0 | 0 | 363 | 0.893 | 0.371 |
| cmp_b0209.jpg | ok | 0.135 | 0.000 | 7 | 1 | 456 | 1.000 | 0.539 |
| cmp_b0222.jpg | ok | 0.157 | 0.014 | 12 | 8 | 580 | 0.933 | 0.567 |
| cmp_b0235.jpg | ok | 0.077 | 0.000 | 28 | 2 | 321 | 0.935 | 0.434 |
| cmp_b0248.jpg | ok | 0.143 | 0.000 | 18 | 1 | 645 | 0.947 | 0.449 |
| cmp_b0261.jpg | ok | 0.147 | 0.073 | 17 | 1 | 1139 | 0.967 | 0.761 |
| cmp_b0274.jpg | ok | 0.134 | 0.223 | 28 | 1 | 670 | 0.975 | 0.531 |
| cmp_b0287.jpg | ok | 0.127 | 0.000 | 2 | 6 | 186 | 0.962 | 0.201 |
| cmp_b0300.jpg | ok | 0.119 | 0.000 | 7 | 1 | 239 | 0.908 | 0.212 |
| cmp_b0313.jpg | ok | 0.176 | 0.000 | 10 | 6 | 164 | 0.976 | 0.172 |
| cmp_b0326.jpg | failed | 0.000 | 0.000 | 0 | 0 | 0 | 0.000 | 0.000 |
| cmp_b0339.jpg | ok | 0.142 | 0.000 | 1 | 1 | 313 | 0.888 | 0.398 |
| cmp_b0352.jpg | failed | 0.000 | 0.000 | 0 | 0 | 0 | 0.000 | 0.000 |
| cmp_b0365.jpg | ok | 0.194 | 0.000 | 16 | 3 | 213 | 0.958 | 0.192 |
| cmp_b0378.jpg | ok | 0.190 | 0.000 | 3 | 10 | 122 | 0.984 | 0.164 |
