from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import cv2
import numpy as np


def run_vectorization(
    ortho_path: Path,
    segmentation_result: dict[str, Any],
    lines_result: dict[str, Any],
    geometry_result: dict[str, Any],
    out_dir: Path,
    cfg: dict[str, Any],
    export_cfg: dict[str, Any],
    logger,
) -> dict[str, Any]:
    image = cv2.imread(str(ortho_path))
    if image is None:
        raise RuntimeError(f"Cannot read orthoimage: {ortho_path}")
    height, width = image.shape[:2]

    snap_tol = float(cfg.get("manhattan_snap_tolerance_px", 4.0))
    angle_tol = float(cfg.get("line_angle_tolerance_deg", 10.0))
    regularization_strength = float(cfg.get("regularization_strength", 1.0))
    regularization_strength = max(regularization_strength, 0.1)
    effective_snap_tol = max(snap_tol * regularization_strength, 1e-6)
    effective_angle_tol = max(angle_tol * regularization_strength, 1e-6)
    outline_sw = int(export_cfg.get("stroke_width_outline", 2))
    detail_sw = int(export_cfg.get("stroke_width_detail", 1))
    padding = int(export_cfg.get("padding_px", 0))

    facade_outline = [
        [padding, padding],
        [max(width - padding, padding), padding],
        [max(width - padding, padding), max(height - padding, padding)],
        [padding, max(height - padding, padding)],
    ]

    windows = [
        _regularize_bbox(item, width, height, effective_snap_tol) for item in segmentation_result.get("windows", [])
    ]
    doors = [_regularize_bbox(item, width, height, effective_snap_tol) for item in segmentation_result.get("doors", [])]
    lines = [
        _regularize_line(item, width, height, effective_snap_tol, effective_angle_tol)
        for item in lines_result.get("lines", [])
    ]

    svg_content = _render_svg(
        width=width,
        height=height,
        facade_outline=facade_outline,
        windows=windows,
        doors=doors,
        lines=lines,
        rectification_score=float(geometry_result.get("rectification_score", 0.0)),
        outline_sw=outline_sw,
        detail_sw=detail_sw,
    )
    svg_path = out_dir / "result.svg"
    svg_path.write_text(svg_content, encoding="utf-8")

    preview = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_polygon(preview, facade_outline, thickness=outline_sw)
    for window in windows:
        x, y, w, h = window["bbox"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 0), detail_sw)
    for door in doors:
        x, y, w, h = door["bbox"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 0), detail_sw)
    for line in lines:
        x1, y1, x2, y2 = line["coords"]
        cv2.line(preview, (x1, y1), (x2, y2), (0, 0, 0), detail_sw, cv2.LINE_AA)
    preview_path = out_dir / "vector_preview.png"
    cv2.imwrite(str(preview_path), preview)

    logger.info(
        "Vectorization stage: windows=%d doors=%d lines=%d (snap=%.2f angle=%.2f strength=%.2f)",
        len(windows),
        len(doors),
        len(lines),
        effective_snap_tol,
        effective_angle_tol,
        regularization_strength,
    )
    return {
        "svg_path": str(svg_path),
        "preview_path": str(preview_path),
        "facade_outline": facade_outline,
        "windows": windows,
        "doors": doors,
        "lines": lines,
        "width": int(width),
        "height": int(height),
        "rectification_score": float(geometry_result.get("rectification_score", 0.0)),
    }


def _regularize_bbox(item: dict[str, Any], width: int, height: int, snap_tol: float) -> dict[str, Any]:
    x, y, w, h = item["bbox"]
    x1 = _clamp(_snap(x, snap_tol), 0, width)
    y1 = _clamp(_snap(y, snap_tol), 0, height)
    x2 = _clamp(_snap(x + w, snap_tol), 0, width)
    y2 = _clamp(_snap(y + h, snap_tol), 0, height)
    if x2 <= x1:
        x2 = min(x1 + 1, width)
    if y2 <= y1:
        y2 = min(y1 + 1, height)
    return {
        "id": item["id"],
        "type": item["type"],
        "confidence": item.get("confidence", 0.0),
        "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
    }


def _regularize_line(
    item: dict[str, Any],
    width: int,
    height: int,
    snap_tol: float,
    angle_tol: float,
) -> dict[str, Any]:
    x1, y1, x2, y2 = item["coords"]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle_abs = abs(angle) % 180.0
    if angle_abs > 90.0:
        angle_abs = 180.0 - angle_abs

    if angle_abs <= angle_tol:
        y2 = y1
    elif abs(90.0 - angle_abs) <= angle_tol:
        x2 = x1

    rx1 = _clamp(_snap(x1, snap_tol), 0, width)
    ry1 = _clamp(_snap(y1, snap_tol), 0, height)
    rx2 = _clamp(_snap(x2, snap_tol), 0, width)
    ry2 = _clamp(_snap(y2, snap_tol), 0, height)
    return {
        "coords": [int(rx1), int(ry1), int(rx2), int(ry2)],
        "confidence": float(item.get("confidence", 0.0)),
    }


def _render_svg(
    width: int,
    height: int,
    facade_outline: list[list[int]],
    windows: list[dict[str, Any]],
    doors: list[dict[str, Any]],
    lines: list[dict[str, Any]],
    rectification_score: float,
    outline_sw: int,
    detail_sw: int,
) -> str:
    outline_points = " ".join(f"{x},{y}" for x, y in facade_outline)
    window_xml = []
    for item in windows:
        x, y, w, h = item["bbox"]
        item_id = escape(item["id"])
        conf = float(item.get("confidence", 0.0))
        window_xml.append(
            f'    <rect id="{item_id}" x="{x}" y="{y}" width="{w}" height="{h}" '
            f'stroke="#000" stroke-width="{detail_sw}" fill="none" data-conf="{conf:.4f}" data-type="window"/>'
        )

    door_xml = []
    for item in doors:
        x, y, w, h = item["bbox"]
        item_id = escape(item["id"])
        conf = float(item.get("confidence", 0.0))
        door_xml.append(
            f'    <rect id="{item_id}" x="{x}" y="{y}" width="{w}" height="{h}" '
            f'stroke="#000" stroke-width="{detail_sw}" fill="none" data-conf="{conf:.4f}" data-type="door"/>'
        )

    lines_xml = []
    for idx, item in enumerate(lines, start=1):
        x1, y1, x2, y2 = item["coords"]
        conf = float(item.get("confidence", 0.0))
        lines_xml.append(
            f'    <line id="line_{idx:04d}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#000" stroke-width="{detail_sw}" data-conf="{conf:.4f}"/>'
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'  <rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
            '  <g id="facade_outline">',
            f'    <polygon id="facade_0001" points="{outline_points}" stroke="#000" stroke-width="{outline_sw}" fill="none"/>',
            "  </g>",
            '  <g id="openings_windows">',
            *window_xml,
            "  </g>",
            '  <g id="openings_doors">',
            *door_xml,
            "  </g>",
            '  <g id="lines">',
            *lines_xml,
            "  </g>",
            (
                '  <g id="metadata" '
                f'data-source="pipeline" data-ortho-width="{width}" data-ortho-height="{height}" '
                f'data-units="px" data-rectification-score="{rectification_score:.4f}"/>'
            ),
            "</svg>",
        ]
    )


def _draw_polygon(image: np.ndarray, points: list[list[int]], thickness: int) -> None:
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 0), thickness=thickness)


def _snap(value: float, tol: float) -> int:
    if tol <= 1e-6:
        return int(round(value))
    return int(round(value / tol) * tol)


def _clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))
