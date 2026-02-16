from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any
import xml.etree.ElementTree as ET

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline


WINDOW_LABEL_NAME = "window"
DOOR_LABEL_NAME = "door"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluation on CMP facade real images")
    parser.add_argument(
        "--dataset-dir",
        default="data/cmp_facade_base/base",
        help="Path to CMP base folder with *.jpg and *.png labels",
    )
    parser.add_argument("--outdir-root", default="runs/real_cmp_base", help="Root directory for per-image runs")
    parser.add_argument("--report-dir", default="reports", help="Directory for aggregated report files")
    parser.add_argument(
        "--report-prefix",
        default="cmp_real_eval",
        help="Prefix for generated report files (<prefix>.csv/.json/.md)",
    )
    parser.add_argument("--config", default="configs/mvp.yaml", help="Pipeline YAML config")
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
            metrics = evaluate_single_run(run_dir=run_dir, image_path=image_path)
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)
            metrics = build_empty_metrics(image_path)

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

    summary = summarize(rows)
    summary["source"] = {
        "dataset": "CMP Facade Database (base split)",
        "url": "https://cmp.felk.cvut.cz/~tylecr1/facade/",
        "evaluated_images": len(rows),
        "limit": args.limit,
        "outdir_root": str(outdir_root.resolve()),
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


def build_empty_metrics(image_path: Path) -> dict[str, Any]:
    return {
        "rectification_score": 0.0,
        "runtime_sec": 0.0,
        "gt_windows_count": 0,
        "pred_windows_count": 0,
        "window_count_abs_error": 0,
        "window_iou": 0.0,
        "gt_doors_count": 0,
        "pred_doors_count": 0,
        "door_count_abs_error": 0,
        "door_iou": 0.0,
        "line_count": 0,
        "line_manhattan_ratio": 0.0,
        "image_width": 0,
        "image_height": 0,
        "image_path": str(image_path),
    }


def evaluate_single_run(run_dir: Path, image_path: Path) -> dict[str, Any]:
    quality = read_json(run_dir / "10_geometry" / "quality.json")
    timing = read_json(run_dir / "50_report" / "timing.json")
    windows = read_json(run_dir / "20_segmentation" / "instances_windows.json")
    doors = read_json(run_dir / "20_segmentation" / "instances_doors.json")
    lines = read_json(run_dir / "30_lines" / "lines.json")

    gt_xml = image_path.with_suffix(".xml")
    if not gt_xml.exists():
        raise FileNotFoundError(f"Ground-truth XML is missing for {image_path.name}")

    gt_objects = parse_cmp_xml(gt_xml)
    gt_windows_norm = [obj["bbox_norm"] for obj in gt_objects if obj["label"] == WINDOW_LABEL_NAME]
    gt_doors_norm = [obj["bbox_norm"] for obj in gt_objects if obj["label"] == DOOR_LABEL_NAME]
    pred_windows_norm = bboxes_to_normalized(windows.get("instances", []), windows["image_width"], windows["image_height"])
    pred_doors_norm = bboxes_to_normalized(doors.get("instances", []), doors["image_width"], doors["image_height"])

    mask_size = 1024
    gt_window_mask = normalized_boxes_to_mask(mask_size, gt_windows_norm)
    gt_door_mask = normalized_boxes_to_mask(mask_size, gt_doors_norm)
    pred_window_mask = normalized_boxes_to_mask(mask_size, pred_windows_norm)
    pred_door_mask = normalized_boxes_to_mask(mask_size, pred_doors_norm)

    gt_windows_count = len(gt_windows_norm)
    gt_doors_count = len(gt_doors_norm)
    pred_windows_count = len(pred_windows_norm)
    pred_doors_count = len(pred_doors_norm)

    line_list = lines.get("lines", [])
    line_manhattan_ratio = calc_line_manhattan_ratio(line_list)

    return {
        "rectification_score": float(quality.get("rectification_score", 0.0)),
        "runtime_sec": float(timing.get("total", 0.0)),
        "gt_windows_count": int(gt_windows_count),
        "pred_windows_count": int(pred_windows_count),
        "window_count_abs_error": int(abs(pred_windows_count - gt_windows_count)),
        "window_iou": float(binary_iou(pred_window_mask, gt_window_mask)),
        "gt_doors_count": int(gt_doors_count),
        "pred_doors_count": int(pred_doors_count),
        "door_count_abs_error": int(abs(pred_doors_count - gt_doors_count)),
        "door_iou": float(binary_iou(pred_door_mask, gt_door_mask)),
        "line_count": int(len(line_list)),
        "line_manhattan_ratio": float(line_manhattan_ratio),
        "image_width": int(windows["image_width"]),
        "image_height": int(windows["image_height"]),
        "image_path": str(image_path),
    }


def bboxes_to_normalized(instances: list[dict[str, Any]], width: int, height: int) -> list[list[float]]:
    out: list[list[float]] = []
    width = max(width, 1)
    height = max(height, 1)
    for inst in instances:
        x, y, w, h = inst["bbox"]
        x1 = np.clip(float(x) / width, 0.0, 1.0)
        y1 = np.clip(float(y) / height, 0.0, 1.0)
        x2 = np.clip(float(x + w) / width, 0.0, 1.0)
        y2 = np.clip(float(y + h) / height, 0.0, 1.0)
        if x2 > x1 and y2 > y1:
            out.append([x1, y1, x2, y2])
    return out


def normalized_boxes_to_mask(size: int, boxes: list[list[float]]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    for x1n, y1n, x2n, y2n in boxes:
        x1 = int(np.clip(round(x1n * size), 0, size))
        y1 = int(np.clip(round(y1n * size), 0, size))
        x2 = int(np.clip(round(x2n * size), 0, size))
        y2 = int(np.clip(round(y2n * size), 0, size))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_bin = pred.astype(bool)
    gt_bin = gt.astype(bool)
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def calc_line_manhattan_ratio(lines: list[dict[str, Any]], tolerance_deg: float = 10.0) -> float:
    if not lines:
        return 0.0
    valid = 0
    for line in lines:
        x1, y1, x2, y2 = line["coords"]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        if angle <= tolerance_deg or abs(90.0 - angle) <= tolerance_deg:
            valid += 1
    return float(valid / len(lines))


def parse_cmp_xml(xml_path: Path) -> list[dict[str, Any]]:
    text = xml_path.read_text(encoding="utf-8").strip()
    wrapped = f"<root>{text}</root>"
    root = ET.fromstring(wrapped)

    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        label_name = (obj.findtext("labelname") or "").strip().lower()
        if label_name not in (WINDOW_LABEL_NAME, DOOR_LABEL_NAME):
            continue

        x_nodes = obj.findall("points/x")
        y_nodes = obj.findall("points/y")
        if len(x_nodes) < 2 or len(y_nodes) < 2:
            continue

        x_vals = sorted(float(node.text.strip()) for node in x_nodes if node.text)
        y_vals = sorted(float(node.text.strip()) for node in y_nodes if node.text)
        if len(x_vals) < 2 or len(y_vals) < 2:
            continue

        x1 = float(np.clip(x_vals[0], 0.0, 1.0))
        x2 = float(np.clip(x_vals[-1], 0.0, 1.0))
        y1 = float(np.clip(y_vals[0], 0.0, 1.0))
        y2 = float(np.clip(y_vals[-1], 0.0, 1.0))
        if x2 <= x1 or y2 <= y1:
            continue

        objects.append({"label": label_name, "bbox_norm": [x1, y1, x2, y2]})
    return objects


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return {
            "status": "failed",
            "num_ok": 0,
            "num_failed": len(rows),
        }

    window_nonempty = [row for row in ok_rows if int(row["gt_windows_count"]) > 0]
    door_nonempty = [row for row in ok_rows if int(row["gt_doors_count"]) > 0]
    door_presence_rows = door_nonempty
    door_presence_recall = (
        mean(1.0 if int(row["pred_doors_count"]) > 0 else 0.0 for row in door_presence_rows)
        if door_presence_rows
        else 1.0
    )

    return {
        "status": "ok",
        "num_ok": len(ok_rows),
        "num_failed": len(rows) - len(ok_rows),
        "avg_rectification_score": mean(row["rectification_score"] for row in ok_rows),
        "avg_runtime_sec": mean(row["runtime_sec"] for row in ok_rows),
        "avg_window_iou": mean(row["window_iou"] for row in ok_rows),
        "avg_door_iou": mean(row["door_iou"] for row in ok_rows),
        "avg_window_iou_nonempty_gt": (
            mean(row["window_iou"] for row in window_nonempty) if window_nonempty else 1.0
        ),
        "avg_door_iou_nonempty_gt": (
            mean(row["door_iou"] for row in door_nonempty) if door_nonempty else 1.0
        ),
        "avg_window_count_abs_error": mean(row["window_count_abs_error"] for row in ok_rows),
        "avg_door_count_abs_error": mean(row["door_count_abs_error"] for row in ok_rows),
        "door_presence_recall_nonempty_gt": door_presence_recall,
        "avg_line_count": mean(row["line_count"] for row in ok_rows),
        "avg_line_manhattan_ratio": mean(row["line_manhattan_ratio"] for row in ok_rows),
    }


def build_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# CMP Real Images Evaluation",
        "",
        "## Summary",
        f"- status: {summary.get('status', 'unknown')}",
        f"- num_ok: {summary.get('num_ok', 0)}",
        f"- num_failed: {summary.get('num_failed', 0)}",
    ]
    if summary.get("status") == "ok":
        lines.extend(
            [
                f"- avg_rectification_score: {summary['avg_rectification_score']:.4f}",
                f"- avg_runtime_sec: {summary['avg_runtime_sec']:.4f}",
                f"- avg_window_iou: {summary['avg_window_iou']:.4f}",
                f"- avg_door_iou: {summary['avg_door_iou']:.4f}",
                f"- avg_window_iou_nonempty_gt: {summary['avg_window_iou_nonempty_gt']:.4f}",
                f"- avg_door_iou_nonempty_gt: {summary['avg_door_iou_nonempty_gt']:.4f}",
                f"- avg_window_count_abs_error: {summary['avg_window_count_abs_error']:.4f}",
                f"- avg_door_count_abs_error: {summary['avg_door_count_abs_error']:.4f}",
                f"- door_presence_recall_nonempty_gt: {summary['door_presence_recall_nonempty_gt']:.4f}",
                f"- avg_line_count: {summary['avg_line_count']:.4f}",
                f"- avg_line_manhattan_ratio: {summary['avg_line_manhattan_ratio']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Per Image",
            "| image | status | rect_score | win_iou | door_iou | win_err | door_err | manhattan | runtime_sec |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['image']} | {row['status']} | {row['rectification_score']:.3f} | {row['window_iou']:.3f} | "
            f"{row['door_iou']:.3f} | {row['window_count_abs_error']} | {row['door_count_abs_error']} | "
            f"{row['line_manhattan_ratio']:.3f} | {row['runtime_sec']:.3f} |"
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
