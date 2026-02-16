from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.io_utils import write_json


def run_segmentation(
    ortho_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    logger,
) -> dict[str, Any]:
    backend = str(cfg.get("backend", "contour_baseline")).strip().lower()
    if backend in {"yolo_seg", "yolo", "ultralytics_yolo_seg"}:
        try:
            return _run_yolo_inference(ortho_path, out_dir, cfg, logger, task="seg")
        except Exception as exc:  # noqa: BLE001
            if not bool(cfg.get("allow_fallback", True)):
                raise
            logger.warning("YOLO segmentation failed, fallback to contour baseline: %s", exc)
            result = _run_contour_segmentation(ortho_path, out_dir, cfg, logger)
            windows_meta = _read_json_safe(out_dir / "instances_windows.json")
            windows_meta["fallback_reason"] = str(exc)
            write_json(out_dir / "instances_windows.json", windows_meta)
            return result
    if backend in {"yolo_det", "ultralytics_yolo_det"}:
        try:
            return _run_yolo_inference(ortho_path, out_dir, cfg, logger, task="det")
        except Exception as exc:  # noqa: BLE001
            if not bool(cfg.get("allow_fallback", True)):
                raise
            logger.warning("YOLO segmentation failed, fallback to contour baseline: %s", exc)
            result = _run_contour_segmentation(ortho_path, out_dir, cfg, logger)
            windows_meta = _read_json_safe(out_dir / "instances_windows.json")
            windows_meta["fallback_reason"] = str(exc)
            write_json(out_dir / "instances_windows.json", windows_meta)
            return result
    return _run_contour_segmentation(ortho_path, out_dir, cfg, logger)


def _run_contour_segmentation(
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

        entity = {
            "bbox": [int(x), int(y), int(w), int(h)],
            "polygon": _rect_to_polygon(x, y, w, h),
            "confidence": round(float(confidence), 4),
            "area_ratio": round(float(area_ratio), 6),
            "source": "contour",
        }

        bottom_norm = (y + h) / max(float(height), 1.0)
        if aspect >= door_aspect_min and bottom_norm >= door_bottom_zone:
            doors.append(entity)
            continue
        if window_aspect_min <= aspect <= window_aspect_max:
            windows.append(entity)

    logger.info("Segmentation stage (contour): windows=%d doors=%d", len(windows), len(doors))
    return _save_segmentation_outputs(image, out_dir, windows, doors, backend="contour_baseline")


def _run_yolo_inference(
    ortho_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    logger,
    task: str,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ultralytics is unavailable: {exc}") from exc

    image = cv2.imread(str(ortho_path))
    if image is None:
        raise RuntimeError(f"Cannot read orthoimage: {ortho_path}")
    height, width = image.shape[:2]

    model_path = str(cfg.get("model_path") or cfg.get("weights") or "").strip()
    if not model_path:
        raise RuntimeError("YOLO backend requires `segmentation.model_path`")
    model = YOLO(model_path)

    min_confidence = float(cfg.get("min_confidence", 0.25))
    iou_threshold = float(cfg.get("iou_threshold", 0.5))
    imgsz = int(cfg.get("imgsz", 640))
    max_det = int(cfg.get("max_det", 400))
    device = str(cfg.get("device", "cpu"))

    predictions = model.predict(
        source=str(ortho_path),
        conf=min_confidence,
        iou=iou_threshold,
        imgsz=imgsz,
        max_det=max_det,
        device=device,
        verbose=False,
        retina_masks=(task == "seg"),
    )
    if not predictions:
        raise RuntimeError("YOLO predict returned no results")

    pred = predictions[0]
    boxes = pred.boxes
    masks_xy = pred.masks.xy if (task == "seg" and pred.masks is not None) else None
    if boxes is None or len(boxes) == 0:
        logger.info("Segmentation stage (yolo): windows=0 doors=0")
        return _save_segmentation_outputs(image, out_dir, [], [], backend="yolo_seg")

    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(np.int64)

    windows: list[dict[str, Any]] = []
    doors: list[dict[str, Any]] = []
    min_area_ratio = float(cfg.get("min_area_ratio", 0.0008))
    max_area_ratio = float(cfg.get("max_area_ratio", 0.35))
    image_area = float(width * height)

    for idx, (bbox_xyxy, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
        if cls_id not in (0, 1):
            continue
        x1, y1, x2, y2 = bbox_xyxy.tolist()
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width - 1))
        y2 = max(0, min(int(round(y2)), height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        bw = x2 - x1
        bh = y2 - y1
        area_ratio = float((bw * bh) / max(image_area, 1.0))
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        polygon = _rect_to_polygon(x1, y1, bw, bh)
        if masks_xy is not None and idx < len(masks_xy):
            poly_raw = masks_xy[idx]
            if poly_raw is not None and len(poly_raw) >= 3:
                polygon = [
                    [int(np.clip(round(point[0]), 0, width - 1)), int(np.clip(round(point[1]), 0, height - 1))]
                    for point in poly_raw
                ]

        entity = {
            "bbox": [int(x1), int(y1), int(bw), int(bh)],
            "polygon": polygon,
            "confidence": round(float(conf), 4),
            "area_ratio": round(float(area_ratio), 6),
            "source": f"yolo_{task}",
        }
        if int(cls_id) == 0:
            windows.append(entity)
        else:
            doors.append(entity)

    logger.info("Segmentation stage (yolo_%s): windows=%d doors=%d", task, len(windows), len(doors))
    return _save_segmentation_outputs(image, out_dir, windows, doors, backend=f"yolo_{task}")


def _save_segmentation_outputs(
    image: np.ndarray,
    out_dir: Path,
    windows: list[dict[str, Any]],
    doors: list[dict[str, Any]],
    backend: str,
) -> dict[str, Any]:
    height, width = image.shape[:2]
    windows = sorted(windows, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    doors = sorted(doors, key=lambda item: (item["bbox"][1], item["bbox"][0]))

    for idx, item in enumerate(windows, start=1):
        item["id"] = f"window_{idx:04d}"
        item["type"] = "window"
    for idx, item in enumerate(doors, start=1):
        item["id"] = f"door_{idx:04d}"
        item["type"] = "door"

    windows_payload = {
        "format_version": "mvp-0.1",
        "backend": backend,
        "image_width": width,
        "image_height": height,
        "instances": windows,
    }
    doors_payload = {
        "format_version": "mvp-0.1",
        "backend": backend,
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
    return [[int(x), int(y)], [int(x + w), int(y)], [int(x + w), int(y + h)], [int(x), int(y + h)]]


def _read_json_safe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))
