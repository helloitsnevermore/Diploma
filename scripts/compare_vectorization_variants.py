from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vectorization variant reports on CMP vector-eval summaries")
    parser.add_argument(
        "--entry",
        action="append",
        required=True,
        help="Entry in format <label>=<summary_json_path>; pass multiple times",
    )
    parser.add_argument("--report-dir", default="reports", help="Output directory for comparison report")
    parser.add_argument("--tag", default="cmp_vector_variant_compare", help="Output file prefix")
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
            "max avg_openings_iou_nonempty_gt, then min avg_window_count_abs_error, then min avg_door_count_abs_error, "
            "then max avg_vector_line_manhattan_ratio, then min avg_runtime_sec"
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
    avg_window_iou_nonempty = float(summary.get("avg_window_iou_nonempty_gt", summary.get("avg_window_iou", 0.0)))
    avg_door_iou_nonempty = float(summary.get("avg_door_iou_nonempty_gt", summary.get("avg_door_iou", 0.0)))
    return {
        "label": label,
        "path": str(path),
        "status": str(summary.get("status", "unknown")),
        "num_ok": int(summary.get("num_ok", 0)),
        "num_failed": int(summary.get("num_failed", 0)),
        "avg_window_iou_nonempty_gt": avg_window_iou_nonempty,
        "avg_door_iou_nonempty_gt": avg_door_iou_nonempty,
        "avg_openings_iou_nonempty_gt": float((avg_window_iou_nonempty + avg_door_iou_nonempty) / 2.0),
        "avg_window_count_abs_error": float(summary.get("avg_window_count_abs_error", 0.0)),
        "avg_door_count_abs_error": float(summary.get("avg_door_count_abs_error", 0.0)),
        "avg_vector_line_manhattan_ratio": float(summary.get("avg_vector_line_manhattan_ratio", 0.0)),
        "avg_vector_line_count": float(summary.get("avg_vector_line_count", 0.0)),
        "avg_runtime_sec": float(summary.get("avg_runtime_sec", 0.0)),
    }


def select_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("status") == "ok" and row.get("num_ok", 0) > 0]
    if not candidates:
        return None

    def key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(row["avg_openings_iou_nonempty_gt"]),
            -float(row["avg_window_count_abs_error"]),
            -float(row["avg_door_count_abs_error"]),
            float(row["avg_vector_line_manhattan_ratio"]),
            -float(row["avg_runtime_sec"]),
        )

    return max(candidates, key=key)


def build_markdown(rows: list[dict[str, Any]], winner: dict[str, Any] | None) -> str:
    lines = [
        "# Vectorization Variant Comparison (CMP real images)",
        "",
        "Selection rule:",
        "- max `avg_openings_iou_nonempty_gt` (average of window/door IoU on nonempty GT)",
        "- then min count errors",
        "- then max `avg_vector_line_manhattan_ratio`",
        "- then min `avg_runtime_sec`",
        "",
        "| variant | status | openings_iou_nonempty | win_iou_nonempty | door_iou_nonempty | win_count_err | door_count_err | line_manhattan | line_count | runtime_sec |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['status']} | {row['avg_openings_iou_nonempty_gt']:.4f} | "
            f"{row['avg_window_iou_nonempty_gt']:.4f} | "
            f"{row['avg_door_iou_nonempty_gt']:.4f} | {row['avg_window_count_abs_error']:.3f} | "
            f"{row['avg_door_count_abs_error']:.3f} | {row['avg_vector_line_manhattan_ratio']:.4f} | "
            f"{row['avg_vector_line_count']:.2f} | {row['avg_runtime_sec']:.4f} |"
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
