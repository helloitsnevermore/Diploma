from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare geometry backends on the same input")
    parser.add_argument("--input", required=True, help="Video file or images directory")
    parser.add_argument("--outdir-root", default="runs/geometry_compare", help="Root for run outputs")
    parser.add_argument("--report-dir", default="reports", help="Output directory for report files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Config path; pass multiple times for comparison",
    )
    parser.add_argument(
        "--tag",
        default="geometry_compare",
        help="Report file prefix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir_root = Path(args.outdir_root)
    outdir_root.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, config_path in enumerate(args.config, start=1):
        config = Path(config_path)
        run_name = f"{idx:02d}_{config.stem}"
        run_dir = outdir_root / run_name
        status = "ok"
        error = ""
        result: dict[str, Any] = {}
        try:
            result = run_pipeline(
                input_path=args.input,
                outdir=run_dir,
                config_path=config,
                seed=args.seed,
                device="cpu",
                debug=False,
            )
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error = str(exc)

        quality = read_json_safe(run_dir / "10_geometry" / "quality.json")
        timing = read_json_safe(run_dir / "50_report" / "timing.json")
        rows.append(
            {
                "config": str(config),
                "run_dir": str(run_dir),
                "status": status,
                "error": error,
                "backend": quality.get("backend", "n/a"),
                "rectification_score": float(quality.get("rectification_score", 0.0)),
                "manhattan_ratio": float(quality.get("manhattan_ratio", 0.0)),
                "coverage": float(quality.get("coverage", 0.0)),
                "rotation_deg": float(quality.get("rotation_deg", 0.0)),
                "reprojection_error": float(quality.get("reprojection_error", 0.0)),
                "num_attempts": int(quality.get("num_attempts", 0)),
                "total_time_sec": float(timing.get("total", 0.0)),
                "geometry_time_sec": float(timing.get("10_geometry", 0.0)),
                "svg_path": str(result.get("vector", {}).get("svg_path", "")),
            }
        )

    json_path = report_dir / f"{args.tag}.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path = report_dir / f"{args.tag}.md"
    md_path.write_text(build_markdown(rows), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def build_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Geometry Backend Comparison",
        "",
        "| config | status | backend | rect_score | manhattan | coverage | attempts | geom_sec | total_sec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{Path(row['config']).name}` | {row['status']} | {row['backend']} | "
            f"{row['rectification_score']:.3f} | {row['manhattan_ratio']:.3f} | {row['coverage']:.3f} | "
            f"{row['num_attempts']} | {row['geometry_time_sec']:.3f} | {row['total_time_sec']:.3f} |"
        )
        if row["error"]:
            lines.append(f"- error (`{Path(row['config']).name}`): {row['error']}")
    return "\n".join(lines) + "\n"


def read_json_safe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
