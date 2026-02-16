# Export Variant Comparison (CMP real images)

Selection rule:
- max `expected_outputs_count`
- then max `export_valid_rate`
- then min `avg_runtime_sec`
- then min `avg_total_export_size_kb`

| variant | status | outputs_count | svg/dxf/pdf | export_valid_rate | svg_valid | dxf_valid | pdf_valid | runtime_sec | total_size_kb |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| full | ok | 3 | 1/1/1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3804 | 206.58 |
| no_pdf | ok | 2 | 1/1/0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3592 | 115.67 |
| svg_only | ok | 1 | 1/0/0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.3722 | 61.00 |

Winner: `full`

Source summaries:
- `full`: `reports\cmp_export_eval_full30_summary.json`
- `no_pdf`: `reports\cmp_export_eval_no_pdf30_summary.json`
- `svg_only`: `reports\cmp_export_eval_svg_only30_summary.json`
