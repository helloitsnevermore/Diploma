from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def ensure_stage_directories(outdir: Path) -> dict[str, Path]:
    stage_dirs = {
        "00_input": outdir / "00_input",
        "10_geometry": outdir / "10_geometry",
        "20_segmentation": outdir / "20_segmentation",
        "30_lines": outdir / "30_lines",
        "40_vector": outdir / "40_vector",
        "50_report": outdir / "50_report",
    }
    for path in stage_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    (stage_dirs["00_input"] / "frames").mkdir(parents=True, exist_ok=True)
    return stage_dirs


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_input_kind(input_path: Path) -> str:
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        return "video"
    if input_path.is_dir():
        return "images_dir"
    if input_path.is_file() and input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return "image_file"
    raise ValueError(f"Unsupported input path: {input_path}")


def _list_images(input_path: Path, image_extensions: list[str]) -> list[Path]:
    ext_set = {ext.lower() for ext in image_extensions}
    if input_path.is_file():
        return [input_path]
    candidates = []
    for item in sorted(input_path.iterdir()):
        if item.is_file() and item.suffix.lower() in ext_set:
            candidates.append(item)
    return candidates


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _manifest_matches(current: dict[str, Any], expected: dict[str, Any]) -> bool:
    keys = ("input", "kind", "fps", "max_frames")
    return all(current.get(key) == expected.get(key) for key in keys)


def extract_or_collect_frames(
    input_path: Path,
    out_frames_dir: Path,
    input_cfg: dict[str, Any],
    logger,
) -> tuple[list[Path], dict[str, Any]]:
    fps = float(input_cfg.get("fps", 2.0))
    max_frames = int(input_cfg.get("max_frames", 120))
    image_extensions = list(input_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"]))
    manifest_path = out_frames_dir.parent / "manifest.json"

    kind = detect_input_kind(input_path)
    expected_manifest = {
        "input": str(input_path.resolve()),
        "kind": kind,
        "fps": fps,
        "max_frames": max_frames,
    }

    if manifest_path.exists():
        cached = read_json(manifest_path)
        if _manifest_matches(cached, expected_manifest):
            cached_frames = [out_frames_dir / item for item in cached.get("frames", [])]
            if cached_frames and all(path.exists() for path in cached_frames):
                logger.info("Input stage cache hit: reuse %d frames", len(cached_frames))
                return cached_frames, cached

    for old_file in out_frames_dir.glob("*"):
        if old_file.is_file():
            old_file.unlink()

    if kind == "video":
        frames = _extract_video_frames(input_path, out_frames_dir, fps, max_frames, logger)
    else:
        frames = _collect_image_frames(input_path, out_frames_dir, max_frames, image_extensions, logger)

    manifest = {
        **expected_manifest,
        "frames": [frame.name for frame in frames],
        "num_frames": len(frames),
    }
    _write_manifest(manifest_path, manifest)
    return frames, manifest


def _extract_video_frames(
    video_path: Path,
    out_frames_dir: Path,
    fps: float,
    max_frames: int,
    logger,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 1e-6:
        native_fps = 25.0
    step = max(int(round(native_fps / max(fps, 1e-3))), 1)

    logger.info(
        "Extracting video frames: native_fps=%.2f target_fps=%.2f step=%d max_frames=%d",
        native_fps,
        fps,
        step,
        max_frames,
    )

    saved_frames: list[Path] = []
    frame_idx = 0
    keep_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            out_path = out_frames_dir / f"frame_{keep_idx:05d}.png"
            cv2.imwrite(str(out_path), frame)
            saved_frames.append(out_path)
            keep_idx += 1
            if keep_idx >= max_frames:
                break
        frame_idx += 1

    cap.release()
    if not saved_frames:
        raise RuntimeError("No frames extracted from video")
    return saved_frames


def _collect_image_frames(
    input_path: Path,
    out_frames_dir: Path,
    max_frames: int,
    image_extensions: list[str],
    logger,
) -> list[Path]:
    images = _list_images(input_path, image_extensions)
    if not images:
        raise RuntimeError(f"No supported images found in: {input_path}")

    selected = images[:max_frames]
    logger.info("Collecting %d images (of %d found)", len(selected), len(images))

    out_paths: list[Path] = []
    for idx, src in enumerate(selected):
        dst = out_frames_dir / f"frame_{idx:05d}.png"
        if src.suffix.lower() == ".png":
            shutil.copyfile(src, dst)
        else:
            image = cv2.imread(str(src))
            if image is None:
                continue
            cv2.imwrite(str(dst), image)
        out_paths.append(dst)

    if not out_paths:
        raise RuntimeError("Image collection produced zero readable frames")
    return out_paths

