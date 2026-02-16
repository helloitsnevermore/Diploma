from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2

from src.io_utils import write_json


def run_lines(
    ortho_path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    logger,
) -> dict[str, Any]:
    image = cv2.imread(str(ortho_path))
    if image is None:
        raise RuntimeError(f"Cannot read orthoimage: {ortho_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    backend = str(cfg.get("backend", "lsd")).strip().lower()
    min_length = float(cfg.get("min_length_px", 30))
    max_lines = int(cfg.get("max_lines", 1500))
    allow_fallback = bool(cfg.get("allow_fallback", True))

    used_backend = backend
    if backend in {"lsd", "line_segment_detector"}:
        lines = _detect_lines_lsd(gray, min_length)
        if not lines and allow_fallback:
            lines = _detect_lines_hough(gray, min_length, cfg)
            if lines:
                used_backend = "hough_fallback"
    elif backend in {"hough", "houghp", "hough_lines_p"}:
        lines = _detect_lines_hough(gray, min_length, cfg)
        used_backend = "hough"
    elif backend in {"auto", "lsd_or_hough"}:
        lines_lsd = _detect_lines_lsd(gray, min_length)
        lines_hough = _detect_lines_hough(gray, min_length, cfg)
        ratio_lsd = _calc_manhattan_ratio(lines_lsd)
        ratio_hough = _calc_manhattan_ratio(lines_hough)
        if ratio_hough > ratio_lsd:
            lines = lines_hough
            used_backend = "hough_auto"
        else:
            lines = lines_lsd if lines_lsd else lines_hough
            used_backend = "lsd_auto" if lines_lsd else "hough_auto"
    else:
        raise RuntimeError(f"Unknown lines backend: {backend}")

    lines.sort(key=lambda item: item["confidence"], reverse=True)
    lines = lines[:max_lines]

    preview = image.copy()
    for segment in lines:
        x1, y1, x2, y2 = segment["coords"]
        cv2.line(preview, (x1, y1), (x2, y2), (255, 120, 0), 1, cv2.LINE_AA)

    preview_path = out_dir / "lines_preview.png"
    cv2.imwrite(str(preview_path), preview)

    payload = {
        "format_version": "mvp-0.1",
        "backend": used_backend,
        "requested_backend": backend,
        "num_lines": len(lines),
        "lines": lines,
    }
    lines_path = out_dir / "lines.json"
    write_json(lines_path, payload)

    logger.info("Lines stage (%s): %d segments", used_backend, len(lines))
    return {
        "lines": lines,
        "backend": used_backend,
        "lines_path": str(lines_path),
        "preview_path": str(preview_path),
    }


def _detect_lines_lsd(gray: Any, min_length: float) -> list[dict[str, Any]]:
    if not hasattr(cv2, "createLineSegmentDetector"):
        return []

    detector = cv2.createLineSegmentDetector(0)
    detected = detector.detect(gray)[0]
    if detected is None:
        return []

    h, w = gray.shape
    diag = max(math.hypot(w, h), 1.0)
    lines = []
    for item in detected:
        x1, y1, x2, y2 = item[0]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        confidence = min(length / diag, 1.0)
        lines.append(
            {
                "coords": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "length": round(length, 3),
                "angle_deg": round(angle, 3),
                "confidence": round(confidence, 4),
                "source": "lsd",
            }
        )
    return lines


def _detect_lines_hough(gray: Any, min_length: float, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    canny_low = int(cfg.get("hough_canny_low", 50))
    canny_high = int(cfg.get("hough_canny_high", 150))
    threshold = int(cfg.get("hough_threshold", 80))
    max_gap = int(cfg.get("hough_max_line_gap_px", 8))
    min_length_hough = int(cfg.get("hough_min_line_length_px", min_length))
    min_length_hough = max(min_length_hough, 10)
    canny_high = max(canny_high, canny_low + 1)
    threshold = max(threshold, 1)
    max_gap = max(max_gap, 0)

    edges = cv2.Canny(gray, canny_low, canny_high)
    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=threshold,
        minLineLength=min_length_hough,
        maxLineGap=max_gap,
    )
    if detected is None:
        return []

    h, w = gray.shape
    diag = max(math.hypot(w, h), 1.0)
    lines = []
    for item in detected:
        x1, y1, x2, y2 = item[0]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        confidence = min(length / diag, 1.0)
        lines.append(
            {
                "coords": [int(x1), int(y1), int(x2), int(y2)],
                "length": round(length, 3),
                "angle_deg": round(angle, 3),
                "confidence": round(confidence, 4),
                "source": "hough",
            }
        )
    return lines


def _calc_manhattan_ratio(lines: list[dict[str, Any]], tolerance_deg: float = 10.0) -> float:
    if not lines:
        return 0.0
    valid = 0
    for line in lines:
        angle = abs(float(line.get("angle_deg", 0.0))) % 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        if angle <= tolerance_deg or abs(90.0 - angle) <= tolerance_deg:
            valid += 1
    return float(valid / len(lines))
