from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import load_config
from src.export import run_export
from src.geometry import run_geometry
from src.io_utils import ensure_stage_directories, extract_or_collect_frames, write_json
from src.lines import run_lines
from src.logging_utils import create_logger
from src.segmentation import run_segmentation
from src.vectorize import run_vectorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Facade vectorization MVP pipeline")
    parser.add_argument("--input", required=True, help="Path to input video or image directory/file")
    parser.add_argument("--outdir", required=True, help="Directory for output artifacts")
    parser.add_argument("--config", default="configs/mvp.yaml", help="Path to YAML config")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device hint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def run_pipeline(
    input_path: str | Path,
    outdir: str | Path,
    config_path: str | Path,
    device: str = "cpu",
    seed: int = 42,
    debug: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage_dirs = ensure_stage_directories(outdir)

    logger = create_logger(outdir / "pipeline.log", debug=debug)
    logger.info("Pipeline start: input=%s outdir=%s config=%s device=%s", input_path, outdir, config_path, device)

    config = load_config(config_path)
    _set_seed(seed)
    logger.info("Seed set to %d", seed)

    timings: dict[str, float] = {}
    stage_results: dict[str, Any] = {"status": "ok"}
    started_at = time.perf_counter()

    try:
        t0 = time.perf_counter()
        frames, manifest = extract_or_collect_frames(
            input_path=input_path,
            out_frames_dir=stage_dirs["00_input"] / "frames",
            input_cfg=config["input"],
            logger=logger,
        )
        timings["00_input"] = time.perf_counter() - t0
        stage_results["manifest"] = manifest

        t0 = time.perf_counter()
        geometry_result = run_geometry(
            frame_paths=frames,
            out_dir=stage_dirs["10_geometry"],
            cfg=config["geometry"],
            logger=logger,
        )
        timings["10_geometry"] = time.perf_counter() - t0
        stage_results["geometry"] = geometry_result

        ortho_path = Path(geometry_result["facade_ortho_path"])

        t0 = time.perf_counter()
        segmentation_result = run_segmentation(
            ortho_path=ortho_path,
            out_dir=stage_dirs["20_segmentation"],
            cfg=config["segmentation"],
            logger=logger,
        )
        timings["20_segmentation"] = time.perf_counter() - t0
        stage_results["segmentation"] = segmentation_result

        t0 = time.perf_counter()
        lines_result = run_lines(
            ortho_path=ortho_path,
            out_dir=stage_dirs["30_lines"],
            cfg=config["lines"],
            logger=logger,
        )
        timings["30_lines"] = time.perf_counter() - t0
        stage_results["lines"] = lines_result

        t0 = time.perf_counter()
        vector_result = run_vectorization(
            ortho_path=ortho_path,
            segmentation_result=segmentation_result,
            lines_result=lines_result,
            geometry_result=geometry_result,
            out_dir=stage_dirs["40_vector"],
            cfg=config["vectorize"],
            export_cfg=config["export"],
            logger=logger,
        )
        timings["40_vectorize"] = time.perf_counter() - t0
        stage_results["vector"] = vector_result

        t0 = time.perf_counter()
        export_result = run_export(
            vector_result=vector_result,
            out_dir=stage_dirs["40_vector"],
            cfg=config["export"],
            logger=logger,
        )
        timings["50_export"] = time.perf_counter() - t0
        stage_results["export"] = export_result

    except Exception as exc:  # noqa: BLE001
        stage_results["status"] = "failed"
        stage_results["error"] = str(exc)
        logger.exception("Pipeline failed: %s", exc)
        _write_reports(stage_dirs["50_report"], timings, stage_results, status="failed")
        raise

    timings["total"] = time.perf_counter() - started_at
    _write_reports(stage_dirs["50_report"], timings, stage_results, status="ok")
    logger.info("Pipeline finished successfully in %.2f sec", timings["total"])
    return stage_results


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _write_reports(
    report_dir: Path,
    timings: dict[str, float],
    stage_results: dict[str, Any],
    status: str,
) -> None:
    write_json(
        report_dir / "timing.json",
        {key: round(value, 4) for key, value in timings.items()},
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_text = _build_report_markdown(stage_results, timings, status)
    (report_dir / "report.md").write_text(report_text, encoding="utf-8")


def _build_report_markdown(stage_results: dict[str, Any], timings: dict[str, float], status: str) -> str:
    lines = [
        "# Pipeline Report",
        "",
        f"- status: **{status}**",
        "",
        "## Timings (sec)",
    ]
    if timings:
        for stage, value in timings.items():
            lines.append(f"- {stage}: {value:.3f}")
    else:
        lines.append("- no timings recorded")

    lines.append("")
    lines.append("## Summary")
    if status == "ok":
        geometry = stage_results.get("geometry", {})
        segmentation = stage_results.get("segmentation", {})
        lines_res = stage_results.get("lines", {})
        vector = stage_results.get("vector", {})
        lines.extend(
            [
                f"- rectification_score: {geometry.get('rectification_score', 0.0):.4f}",
                f"- windows: {len(segmentation.get('windows', []))}",
                f"- doors: {len(segmentation.get('doors', []))}",
                f"- line_segments: {len(lines_res.get('lines', []))}",
                f"- svg: {vector.get('svg_path', 'n/a')}",
            ]
        )
    else:
        lines.append(f"- error: {stage_results.get('error', 'unknown')}")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        outdir=args.outdir,
        config_path=args.config,
        device=args.device,
        seed=args.seed,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()

