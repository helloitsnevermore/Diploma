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
    min_length = float(cfg.get("min_length_px", 30))
    max_lines = int(cfg.get("max_lines", 1500))

    lines = _detect_lines_lsd(gray, min_length)
    if not lines:
        lines = _detect_lines_hough(gray, min_length)

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
        "num_lines": len(lines),
        "lines": lines,
    }
    lines_path = out_dir / "lines.json"
    write_json(lines_path, payload)

    logger.info("Lines stage: %d segments", len(lines))
    return {
        "lines": lines,
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


def _detect_lines_hough(gray: Any, min_length: float) -> list[dict[str, Any]]:
    edges = cv2.Canny(gray, 50, 150)
    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=80,
        minLineLength=max(int(min_length), 10),
        maxLineGap=8,
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

