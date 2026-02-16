from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch export validation on CMP facade real images")
    parser.add_argument(
        "--dataset-dir",
        default="data/cmp_facade_base/base",
        help="Path to CMP base folder with *.jpg files",
    )
    parser.add_argument("--outdir-root", default="runs/cmp_export_eval", help="Root directory for per-image runs")
    parser.add_argument("--report-dir", default="reports", help="Directory for aggregated report files")
    parser.add_argument(
        "--report-prefix",
        default="cmp_export_eval",
        help="Prefix for generated report files (<prefix>.csv/.json/.md)",
    )
    parser.add_argument("--config", default="configs/mvp_real_eval.yaml", help="Pipeline YAML config")
    parser.add_argument("--limit", type=int, default=10, help="How many images to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Pipeline seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    outdir_root = Path(args.outdir_root)
    report_dir = Path(args.report_dir)
    outdir_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    export_cfg = config.get("export", {})
    expect_svg = bool(export_cfg.get("write_svg", True))
    expect_dxf = bool(export_cfg.get("write_dxf", True))
    expect_pdf = bool(export_cfg.get("write_pdf_preview", True))

    images = sorted(dataset_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No .jpg files found in {dataset_dir}")

    selected = select_evenly(images, max(1, min(args.limit, len(images))))
    rows: list[dict[str, Any]] = []

    for image_path in selected:
        run_name = image_path.stem
        run_dir = outdir_root / run_name
        status = "ok"
        error = ""

        try:
            run_pipeline(
                input_path=image_path,
                outdir=run_dir,
                config_path=args.config,
                device="cpu",
                seed=args.seed,
                debug=args.debug,
            )
            metrics = evaluate_single_run(
                run_dir=run_dir,
                expect_svg=expect_svg,
                expect_dxf=expect_dxf,
                expect_pdf=expect_pdf,
            )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
            metrics = build_empty_metrics(expect_svg=expect_svg, expect_dxf=expect_dxf, expect_pdf=expect_pdf)

        rows.append(
            {
                "image": image_path.name,
                "status": status,
                "error": error,
                **metrics,
            }
        )

    csv_path = report_dir / f"{args.report_prefix}.csv"
    write_csv(csv_path, rows)

    summary = summarize(rows=rows, expect_svg=expect_svg, expect_dxf=expect_dxf, expect_pdf=expect_pdf)
    summary["source"] = {
        "dataset": "CMP Facade Database (base split)",
        "url": "https://cmp.felk.cvut.cz/~tylecr1/facade/",
        "evaluated_images": len(rows),
        "limit": args.limit,
        "outdir_root": str(outdir_root.resolve()),
        "config": str(Path(args.config).resolve()),
    }

    json_path = report_dir / f"{args.report_prefix}_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_path = report_dir / f"{args.report_prefix}.md"
    md_path.write_text(build_markdown(rows, summary), encoding="utf-8")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def select_evenly(items: list[Path], limit: int) -> list[Path]:
    if limit >= len(items):
        return items
    idx = np.linspace(0, len(items) - 1, num=limit, dtype=int)
    return [items[i] for i in idx]


def build_empty_metrics(expect_svg: bool, expect_dxf: bool, expect_pdf: bool) -> dict[str, Any]:
    return {
        "runtime_sec": 0.0,
        "svg_expected": bool(expect_svg),
        "dxf_expected": bool(expect_dxf),
        "pdf_expected": bool(expect_pdf),
        "svg_valid": False if expect_svg else True,
        "dxf_valid": False if expect_dxf else True,
        "pdf_valid": False if expect_pdf else True,
        "export_valid_all": False,
        "svg_size_bytes": 0,
        "dxf_size_bytes": 0,
        "pdf_size_bytes": 0,
        "svg_error": "pipeline_failed",
        "dxf_error": "pipeline_failed" if expect_dxf else "",
        "pdf_error": "pipeline_failed" if expect_pdf else "",
    }


def evaluate_single_run(run_dir: Path, expect_svg: bool, expect_dxf: bool, expect_pdf: bool) -> dict[str, Any]:
    timing = read_json(run_dir / "50_report" / "timing.json")
    svg_path = run_dir / "40_vector" / "result.svg"
    dxf_path = run_dir / "40_vector" / "result.dxf"
    pdf_path = run_dir / "40_vector" / "result.pdf"

    svg_check = validate_svg(svg_path, expect_svg)
    dxf_check = validate_dxf(dxf_path, expect_dxf)
    pdf_check = validate_pdf(pdf_path, expect_pdf)

    export_valid_all = bool(svg_check["ok"] and dxf_check["ok"] and pdf_check["ok"])
    return {
        "runtime_sec": float(timing.get("total", 0.0)),
        "svg_expected": bool(expect_svg),
        "dxf_expected": bool(expect_dxf),
        "pdf_expected": bool(expect_pdf),
        "svg_valid": bool(svg_check["ok"]),
        "dxf_valid": bool(dxf_check["ok"]),
        "pdf_valid": bool(pdf_check["ok"]),
        "export_valid_all": export_valid_all,
        "svg_size_bytes": int(file_size(svg_path)),
        "dxf_size_bytes": int(file_size(dxf_path)),
        "pdf_size_bytes": int(file_size(pdf_path)),
        "svg_error": str(svg_check.get("error", "")),
        "dxf_error": str(dxf_check.get("error", "")),
        "pdf_error": str(pdf_check.get("error", "")),
    }


def validate_svg(path: Path, expected: bool) -> dict[str, Any]:
    if not expected:
        return {"ok": True, "error": ""}
    if not path.exists():
        return {"ok": False, "error": "missing"}

    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"xml_parse_error:{exc}"}

    if local_name(root.tag) != "svg":
        return {"ok": False, "error": "root_not_svg"}

    width, height = parse_svg_size(root)
    if width <= 0 or height <= 0:
        return {"ok": False, "error": "invalid_viewbox_or_size"}

    required_groups = {"facade_outline", "openings_windows", "openings_doors", "lines", "metadata"}
    found_groups = set()
    for elem in root.iter():
        if local_name(elem.tag) == "g":
            group_id = elem.attrib.get("id", "").strip()
            if group_id:
                found_groups.add(group_id)

    missing = sorted(required_groups - found_groups)
    if missing:
        return {"ok": False, "error": f"missing_groups:{','.join(missing)}"}

    return {"ok": True, "error": ""}


def validate_dxf(path: Path, expected: bool) -> dict[str, Any]:
    if not expected:
        return {"ok": True, "error": ""}
    if not path.exists():
        return {"ok": False, "error": "missing"}

    text = path.read_text(encoding="utf-8", errors="replace")
    for token in ("SECTION", "ENTITIES", "EOF", "FACADE_OUTLINE"):
        if token not in text:
            return {"ok": False, "error": f"missing_token:{token}"}
    return {"ok": True, "error": ""}


def validate_pdf(path: Path, expected: bool) -> dict[str, Any]:
    if not expected:
        return {"ok": True, "error": ""}
    if not path.exists():
        return {"ok": False, "error": "missing"}
    try:
        with path.open("rb") as f:
            header = f.read(5)
        if header != b"%PDF-":
            return {"ok": False, "error": "invalid_header"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"read_error:{exc}"}
    return {"ok": True, "error": ""}


def parse_svg_size(root: ET.Element) -> tuple[float, float]:
    width = parse_numeric(root.attrib.get("width", ""))
    height = parse_numeric(root.attrib.get("height", ""))
    if width > 0 and height > 0:
        return width, height

    view_box = root.attrib.get("viewBox", "").strip()
    if not view_box:
        return 0.0, 0.0
    parts = [p for p in view_box.replace(",", " ").split() if p]
    if len(parts) != 4:
        return 0.0, 0.0
    try:
        width = float(parts[2])
        height = float(parts[3])
    except ValueError:
        return 0.0, 0.0
    return width, height


def parse_numeric(value: str) -> float:
    if not value:
        return 0.0
    value = value.strip().lower()
    if value.endswith("px"):
        value = value[:-2]
    try:
        return float(value)
    except ValueError:
        return 0.0


def local_name(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except FileNotFoundError:
        return 0


def summarize(rows: list[dict[str, Any]], expect_svg: bool, expect_dxf: bool, expect_pdf: bool) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return {
            "status": "failed",
            "num_ok": 0,
            "num_failed": len(rows),
            "expected_outputs_count": int(expect_svg) + int(expect_dxf) + int(expect_pdf),
            "expect_svg": bool(expect_svg),
            "expect_dxf": bool(expect_dxf),
            "expect_pdf": bool(expect_pdf),
        }

    expected_outputs_count = int(expect_svg) + int(expect_dxf) + int(expect_pdf)
    return {
        "status": "ok",
        "num_ok": len(ok_rows),
        "num_failed": len(rows) - len(ok_rows),
        "expect_svg": bool(expect_svg),
        "expect_dxf": bool(expect_dxf),
        "expect_pdf": bool(expect_pdf),
        "expected_outputs_count": expected_outputs_count,
        "export_valid_rate": mean(1.0 if row["export_valid_all"] else 0.0 for row in ok_rows),
        "svg_valid_rate_expected": mean(1.0 if row["svg_valid"] else 0.0 for row in ok_rows),
        "dxf_valid_rate_expected": (mean(1.0 if row["dxf_valid"] else 0.0 for row in ok_rows) if expect_dxf else 1.0),
        "pdf_valid_rate_expected": (mean(1.0 if row["pdf_valid"] else 0.0 for row in ok_rows) if expect_pdf else 1.0),
        "avg_runtime_sec": mean(float(row["runtime_sec"]) for row in ok_rows),
        "avg_svg_size_kb": mean(float(row["svg_size_bytes"]) / 1024.0 for row in ok_rows),
        "avg_dxf_size_kb": mean(float(row["dxf_size_bytes"]) / 1024.0 for row in ok_rows),
        "avg_pdf_size_kb": mean(float(row["pdf_size_bytes"]) / 1024.0 for row in ok_rows),
        "avg_total_export_size_kb": mean(
            (float(row["svg_size_bytes"]) + float(row["dxf_size_bytes"]) + float(row["pdf_size_bytes"])) / 1024.0
            for row in ok_rows
        ),
    }


def build_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# CMP Export Evaluation",
        "",
        "## Summary",
        f"- status: {summary.get('status', 'unknown')}",
        f"- num_ok: {summary.get('num_ok', 0)}",
        f"- num_failed: {summary.get('num_failed', 0)}",
        f"- expected_outputs_count: {summary.get('expected_outputs_count', 0)}",
        f"- expect_svg: {summary.get('expect_svg', False)}",
        f"- expect_dxf: {summary.get('expect_dxf', False)}",
        f"- expect_pdf: {summary.get('expect_pdf', False)}",
    ]

    if summary.get("status") == "ok":
        lines.extend(
            [
                f"- export_valid_rate: {summary['export_valid_rate']:.4f}",
                f"- svg_valid_rate_expected: {summary['svg_valid_rate_expected']:.4f}",
                f"- dxf_valid_rate_expected: {summary['dxf_valid_rate_expected']:.4f}",
                f"- pdf_valid_rate_expected: {summary['pdf_valid_rate_expected']:.4f}",
                f"- avg_runtime_sec: {summary['avg_runtime_sec']:.4f}",
                f"- avg_svg_size_kb: {summary['avg_svg_size_kb']:.2f}",
                f"- avg_dxf_size_kb: {summary['avg_dxf_size_kb']:.2f}",
                f"- avg_pdf_size_kb: {summary['avg_pdf_size_kb']:.2f}",
                f"- avg_total_export_size_kb: {summary['avg_total_export_size_kb']:.2f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Per Image",
            "| image | status | svg_ok | dxf_ok | pdf_ok | export_ok | runtime_sec | svg_kb | dxf_kb | pdf_kb |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['status']} | {int(bool(row['svg_valid']))} | {int(bool(row['dxf_valid']))} | "
            f"{int(bool(row['pdf_valid']))} | {int(bool(row['export_valid_all']))} | {float(row['runtime_sec']):.3f} | "
            f"{float(row['svg_size_bytes']) / 1024.0:.2f} | {float(row['dxf_size_bytes']) / 1024.0:.2f} | "
            f"{float(row['pdf_size_bytes']) / 1024.0:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
