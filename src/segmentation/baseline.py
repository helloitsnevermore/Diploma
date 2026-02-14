from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from src.io_utils import write_json


def run_segmentation(
    ortho_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    logger,
) -> dict[str, Any]:
    image = cv2.imread(str(ortho_path))
    if image is None:
        raise RuntimeError(f"Cannot read orthoimage: {ortho_path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    min_area_ratio = float(cfg.get("min_area_ratio", 0.0015))
    max_area_ratio = float(cfg.get("max_area_ratio", 0.2))
    window_aspect_min = float(cfg.get("window_aspect_min", 0.35))
    window_aspect_max = float(cfg.get("window_aspect_max", 2.2))
    door_aspect_min = float(cfg.get("door_aspect_min", 1.8))
    door_bottom_zone = float(cfg.get("door_bottom_zone", 0.55))
    min_confidence = float(cfg.get("min_confidence", 0.2))

    windows: list[dict[str, Any]] = []
    doors: list[dict[str, Any]] = []
    image_area = float(width * height)

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / max(image_area, 1.0)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        aspect = h / max(float(w), 1.0)
        rect_area = float(w * h)
        fill_ratio = area / max(rect_area, 1.0)
        confidence = 0.5 * min(1.0, area_ratio / max(min_area_ratio, 1e-6)) + 0.5 * min(fill_ratio, 1.0)
        if confidence < min_confidence:
            continue

        polygon = _rect_to_polygon(x, y, w, h)
        entity = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "polygon": polygon,
            "confidence": round(float(confidence), 4),
            "area_ratio": round(float(area_ratio), 6),
        }

        bottom_norm = (y + h) / max(float(height), 1.0)
        if aspect >= door_aspect_min and bottom_norm >= door_bottom_zone:
            doors.append(entity)
            continue

        if window_aspect_min <= aspect <= window_aspect_max:
            windows.append(entity)

    windows.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    doors.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    for idx, item in enumerate(windows, start=1):
        item["id"] = f"window_{idx:04d}"
        item["type"] = "window"
    for idx, item in enumerate(doors, start=1):
        item["id"] = f"door_{idx:04d}"
        item["type"] = "door"

    windows_payload = {
        "format_version": "mvp-0.1",
        "image_width": width,
        "image_height": height,
        "instances": windows,
    }
    doors_payload = {
        "format_version": "mvp-0.1",
        "image_width": width,
        "image_height": height,
        "instances": doors,
    }
    windows_path = out_dir / "instances_windows.json"
    doors_path = out_dir / "instances_doors.json"
    write_json(windows_path, windows_payload)
    write_json(doors_path, doors_payload)

    preview = image.copy()
    for item in windows:
        x, y, w, h = item["bbox"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 180, 0), 2)
        cv2.putText(preview, item["id"], (x, max(y - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
    for item in doors:
        x, y, w, h = item["bbox"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 220), 2)
        cv2.putText(preview, item["id"], (x, max(y - 4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
    preview_path = out_dir / "masks_preview.png"
    cv2.imwrite(str(preview_path), preview)

    logger.info("Segmentation stage: windows=%d doors=%d", len(windows), len(doors))
    return {
        "windows": windows,
        "doors": doors,
        "windows_path": str(windows_path),
        "doors_path": str(doors_path),
        "preview_path": str(preview_path),
        "image_width": int(width),
        "image_height": int(height),
    }


def _rect_to_polygon(x: int, y: int, w: int, h: int) -> list[list[int]]:
    return [
        [int(x), int(y)],
        [int(x + w), int(y)],
        [int(x + w), int(y + h)],
        [int(x), int(y + h)],
    ]
