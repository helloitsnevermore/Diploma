from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare export variant reports on CMP export-eval summaries")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Entry in format <label>=<summary_json_path>; pass multiple times",
    )
    parser.add_argument("--report-dir", default="reports", help="Output directory for comparison report")
    parser.add_argument("--tag", default="cmp_export_variant_compare", help="Output file prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for raw_entry in args.entry:
        label, summary_path = parse_entry(raw_entry)
        path = Path(summary_path)
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append(build_row(label=label, path=path, summary=summary))

    winner = select_winner(rows)
    result = {
        "winner": winner["label"] if winner else None,
        "selection_rule": (
            "max expected_outputs_count, then max export_valid_rate, "
            "then min avg_runtime_sec, then min avg_total_export_size_kb"
        ),
        "rows": rows,
    }

    json_path = report_dir / f"{args.tag}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    md_path = report_dir / f"{args.tag}.md"
    md_path.write_text(build_markdown(rows=rows, winner=winner), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def parse_entry(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid --entry format: {raw}. Expected <label>=<summary_json_path>")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Invalid --entry format: {raw}. Expected non-empty label and path")
    return label, path


def build_row(label: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path),
        "status": str(summary.get("status", "unknown")),
        "num_ok": int(summary.get("num_ok", 0)),
        "num_failed": int(summary.get("num_failed", 0)),
        "expected_outputs_count": int(summary.get("expected_outputs_count", 0)),
        "expect_svg": bool(summary.get("expect_svg", False)),
        "expect_dxf": bool(summary.get("expect_dxf", False)),
        "expect_pdf": bool(summary.get("expect_pdf", False)),
        "export_valid_rate": float(summary.get("export_valid_rate", 0.0)),
        "svg_valid_rate_expected": float(summary.get("svg_valid_rate_expected", 0.0)),
        "dxf_valid_rate_expected": float(summary.get("dxf_valid_rate_expected", 0.0)),
        "pdf_valid_rate_expected": float(summary.get("pdf_valid_rate_expected", 0.0)),
        "avg_runtime_sec": float(summary.get("avg_runtime_sec", 0.0)),
        "avg_total_export_size_kb": float(summary.get("avg_total_export_size_kb", 0.0)),
    }


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("status") == "ok" and row.get("num_ok", 0) > 0]
    if not candidates:
        return None

    def key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            float(row["expected_outputs_count"]),
            float(row["export_valid_rate"]),
            -float(row["avg_runtime_sec"]),
            -float(row["avg_total_export_size_kb"]),
        )

    return max(candidates, key=key)


def build_markdown(rows: list[dict[str, Any]], winner: dict[str, Any] | None) -> str:
    lines = [
        "# Export Variant Comparison (CMP real images)",
        "",
        "Selection rule:",
        "- max `expected_outputs_count`",
        "- then max `export_valid_rate`",
        "- then min `avg_runtime_sec`",
        "- then min `avg_total_export_size_kb`",
        "",
        "| variant | status | outputs_count | svg/dxf/pdf | export_valid_rate | svg_valid | dxf_valid | pdf_valid | runtime_sec | total_size_kb |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mode = f"{int(row['expect_svg'])}/{int(row['expect_dxf'])}/{int(row['expect_pdf'])}"
        lines.append(
            f"| {row['label']} | {row['status']} | {row['expected_outputs_count']} | {mode} | "
            f"{row['export_valid_rate']:.4f} | {row['svg_valid_rate_expected']:.4f} | "
            f"{row['dxf_valid_rate_expected']:.4f} | {row['pdf_valid_rate_expected']:.4f} | "
            f"{row['avg_runtime_sec']:.4f} | {row['avg_total_export_size_kb']:.2f} |"
        )

    lines.append("")
    lines.append(f"Winner: `{winner['label']}`" if winner else "Winner: n/a (no successful rows)")
    lines.append("")
    lines.append("Source summaries:")
    for row in rows:
        lines.append(f"- `{row['label']}`: `{row['path']}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
