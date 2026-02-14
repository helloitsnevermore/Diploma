from __future__ import annotations

from pathlib import Path
from typing import Any


def run_export(
    vector_result: dict[str, Any],
    out_dir: Path,
    cfg: dict[str, Any],
    logger,
) -> dict[str, Any]:
    outputs: dict[str, str] = {"svg_path": vector_result["svg_path"], "preview_path": vector_result["preview_path"]}

    if bool(cfg.get("write_dxf", True)):
        dxf_path = out_dir / "result.dxf"
        dxf_path.write_text(_to_dxf(vector_result), encoding="utf-8")
        outputs["dxf_path"] = str(dxf_path)

    if bool(cfg.get("write_pdf_preview", True)):
        pdf_path = out_dir / "result.pdf"
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel

            Image.open(vector_result["preview_path"]).convert("RGB").save(pdf_path, "PDF", resolution=300.0)
            outputs["pdf_path"] = str(pdf_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Export stage: failed to generate PDF preview: %s", exc)

    return outputs


def _to_dxf(vector_result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.extend(["0", "SECTION", "2", "ENTITIES"])

    outline = vector_result.get("facade_outline", [])
    if outline:
        lines.extend(_dxf_lwpolyline("FACADE_OUTLINE", outline, closed=True))

    for item in vector_result.get("windows", []):
        x, y, w, h = item["bbox"]
        points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        lines.extend(_dxf_lwpolyline("WINDOWS", points, closed=True))

    for item in vector_result.get("doors", []):
        x, y, w, h = item["bbox"]
        points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        lines.extend(_dxf_lwpolyline("DOORS", points, closed=True))

    for item in vector_result.get("lines", []):
        x1, y1, x2, y2 = item["coords"]
        lines.extend(
            [
                "0",
                "LINE",
                "8",
                "LINES",
                "10",
                f"{float(x1):.6f}",
                "20",
                f"{float(y1):.6f}",
                "30",
                "0.0",
                "11",
                f"{float(x2):.6f}",
                "21",
                f"{float(y2):.6f}",
                "31",
                "0.0",
            ]
        )

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"


def _dxf_lwpolyline(layer: str, points: list[list[int]], closed: bool) -> list[str]:
    out = ["0", "LWPOLYLINE", "8", layer, "90", str(len(points)), "70", "1" if closed else "0"]
    for x, y in points:
        out.extend(["10", f"{float(x):.6f}", "20", f"{float(y):.6f}"])
    return out

