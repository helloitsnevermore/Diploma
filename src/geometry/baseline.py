from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.io_utils import write_json


@dataclass
class RectificationAttempt:
    frame_path: Path
    canny_low: int
    canny_high: int
    score: float
    sharpness: float
    coverage: float
    manhattan_ratio: float
    had_quad: bool
    ortho: np.ndarray


def run_geometry(
    frame_paths: list[Path],
    out_dir: Path,
    cfg: dict[str, Any],
    logger,
) -> dict[str, Any]:
    if not frame_paths:
        raise RuntimeError("Geometry stage received no frames")

    backend = str(cfg.get("backend", "homography_baseline")).strip().lower()
    allow_fallback = bool(cfg.get("allow_fallback", True))
    min_rectification_score = float(cfg.get("min_rectification_score", 0.45))

    pycolmap_error: str | None = None
    if backend in {"pycolmap", "auto", "a1_pycolmap"}:
        min_frames = int(cfg.get("pycolmap_min_frames", 4))
        if len(frame_paths) >= min_frames:
            try:
                return _run_geometry_pycolmap(
                    frame_paths=frame_paths,
                    out_dir=out_dir,
                    cfg=cfg,
                    logger=logger,
                    min_rectification_score=min_rectification_score,
                )
            except Exception as exc:  # noqa: BLE001
                pycolmap_error = str(exc)
                logger.warning("PyCOLMAP geometry attempt failed: %s", exc)
        else:
            pycolmap_error = f"not enough frames for pycolmap: {len(frame_paths)} < {min_frames}"
            logger.info("Skip PyCOLMAP backend: %s", pycolmap_error)

        if backend in {"pycolmap", "a1_pycolmap"} and not allow_fallback:
            raise RuntimeError(f"PyCOLMAP geometry failed and fallback is disabled: {pycolmap_error}")

        logger.info("Geometry fallback: using homography baseline")

    result = _run_geometry_homography(
        frame_paths=frame_paths,
        out_dir=out_dir,
        cfg=cfg,
        logger=logger,
        min_rectification_score=min_rectification_score,
    )
    if pycolmap_error:
        quality = _read_json_safe(out_dir / "quality.json")
        quality["fallback_reason"] = pycolmap_error
        write_json(out_dir / "quality.json", quality)
    return result


def _run_geometry_pycolmap(
    frame_paths: list[Path],
    out_dir: Path,
    cfg: dict[str, Any],
    logger,
    min_rectification_score: float,
) -> dict[str, Any]:
    try:
        import pycolmap  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"pycolmap is unavailable: {exc}") from exc

    frames_dir = frame_paths[0].parent
    if not all(path.parent == frames_dir for path in frame_paths):
        raise RuntimeError("PyCOLMAP backend expects all frames in one directory")

    workspace_root = out_dir / "colmap_workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)

    variants = cfg.get("pycolmap_variants")
    if not variants:
        variants = [
            {
                "name": "v1_fast",
                "frame_step": 2,
                "max_frames": 24,
                "max_image_size": 1600,
                "max_num_features": 4096,
                "num_threads": 8,
            },
            {
                "name": "v2_dense",
                "frame_step": 1,
                "max_frames": 30,
                "max_image_size": 2000,
                "max_num_features": 8192,
                "num_threads": 8,
            },
        ]

    best: dict[str, Any] | None = None
    attempts_log: list[dict[str, Any]] = []
    early_stop_score = float(cfg.get("pycolmap_early_stop_score", 0.72))

    for idx, variant in enumerate(variants, start=1):
        attempt_name = str(variant.get("name", f"attempt_{idx}"))
        frame_step = max(int(variant.get("frame_step", 1)), 1)
        max_frames = max(int(variant.get("max_frames", 24)), 2)
        selected = frame_paths[::frame_step][:max_frames]
        if len(selected) < 2:
            attempts_log.append(
                {
                    "name": attempt_name,
                    "status": "skipped",
                    "reason": f"not enough frames after sampling: {len(selected)}",
                }
            )
            continue

        attempt_dir = workspace_root / f"{idx:02d}_{attempt_name}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        database_path = attempt_dir / "database.db"
        sparse_path = attempt_dir / "sparse"
        sparse_path.mkdir(parents=True, exist_ok=True)
        if database_path.exists():
            database_path.unlink()

        image_names = [path.name for path in selected]
        extraction_options = pycolmap.FeatureExtractionOptions()
        extraction_options.max_image_size = int(variant.get("max_image_size", 1600))
        extraction_options.num_threads = int(variant.get("num_threads", 8))
        extraction_options.use_gpu = bool(variant.get("use_gpu", False))
        extraction_options.sift.max_num_features = int(variant.get("max_num_features", 4096))

        matching_options = pycolmap.FeatureMatchingOptions()
        matching_options.num_threads = int(variant.get("num_threads", 8))
        matching_options.use_gpu = bool(variant.get("use_gpu", False))
        matching_options.max_num_matches = int(variant.get("max_num_matches", 16384))
        matching_options.guided_matching = bool(variant.get("guided_matching", True))

        inc_options = pycolmap.IncrementalPipelineOptions()
        inc_options.num_threads = int(variant.get("num_threads", 8))
        inc_options.min_model_size = int(variant.get("min_model_size", 3))
        inc_options.mapper.abs_pose_min_inlier_ratio = float(
            variant.get("abs_pose_min_inlier_ratio", 0.25)
        )
        inc_options.mapper.init_min_num_inliers = int(variant.get("init_min_num_inliers", 80))
        inc_options.mapper.filter_max_reproj_error = float(variant.get("filter_max_reproj_error", 4.0))
        inc_options.image_names = image_names

        try:
            pycolmap.extract_features(
                str(database_path),
                str(frames_dir),
                image_names=image_names,
                camera_model=str(variant.get("camera_model", "SIMPLE_RADIAL")),
                extraction_options=extraction_options,
            )
            pycolmap.match_exhaustive(
                str(database_path),
                matching_options=matching_options,
            )
            reconstructions = pycolmap.incremental_mapping(
                str(database_path),
                str(frames_dir),
                str(sparse_path),
                options=inc_options,
            )
        except Exception as exc:  # noqa: BLE001
            attempts_log.append(
                {
                    "name": attempt_name,
                    "status": "failed",
                    "reason": f"pycolmap runtime error: {exc}",
                }
            )
            continue

        if not reconstructions:
            attempts_log.append(
                {
                    "name": attempt_name,
                    "status": "failed",
                    "reason": "no reconstruction was produced",
                }
            )
            continue

        reconstruction = max(
            reconstructions.values(),
            key=lambda rec: (int(rec.num_reg_images()), int(rec.num_points3D())),
        )
        if int(reconstruction.num_reg_images()) < 2 or int(reconstruction.num_points3D()) < 100:
            attempts_log.append(
                {
                    "name": attempt_name,
                    "status": "failed",
                    "reason": "insufficient reconstruction size",
                    "num_reg_images": int(reconstruction.num_reg_images()),
                    "num_points3D": int(reconstruction.num_points3D()),
                }
            )
            continue

        points = _collect_points3d(reconstruction)
        plane = _fit_plane_ransac(
            points,
            dist_ratio=float(cfg.get("plane_ransac_dist_ratio", 0.01)),
            max_iters=int(cfg.get("plane_ransac_max_iters", 500)),
        )
        if plane is None:
            attempts_log.append(
                {
                    "name": attempt_name,
                    "status": "failed",
                    "reason": "plane fitting failed",
                    "num_points3D": int(reconstruction.num_points3D()),
                }
            )
            continue

        render = _render_orthomosaic(
            reconstruction=reconstruction,
            frames_dir=frames_dir,
            plane=plane,
            max_output_size=int(cfg.get("pycolmap_max_output_size", 2200)),
            min_output_size=int(cfg.get("pycolmap_min_output_size", 512)),
        )
        if render is None:
            attempts_log.append(
                {
                    "name": attempt_name,
                    "status": "failed",
                    "reason": "orthomosaic rendering failed",
                }
            )
            continue

        ortho = render["ortho"]
        ortho, rotation_deg, manhattan_ratio = _align_to_manhattan(
            ortho,
            tolerance_deg=float(cfg.get("line_angle_tolerance_deg", 10.0)),
        )
        coverage = float(render["coverage"])
        sharpness = _variance_of_laplacian(ortho)
        reprojection_error = float(reconstruction.compute_mean_reprojection_error())
        reg_ratio = float(reconstruction.num_reg_images()) / max(len(selected), 1)
        reproj_score = float(np.exp(-reprojection_error / 3.0))
        sharpness_norm = min(sharpness / 1200.0, 1.0)
        score = float(
            np.clip(
                0.22 * reg_ratio
                + 0.20 * plane["inlier_ratio"]
                + 0.20 * coverage
                + 0.15 * manhattan_ratio
                + 0.13 * sharpness_norm
                + 0.10 * reproj_score,
                0.0,
                1.0,
            )
        )

        attempt_payload = {
            "name": attempt_name,
            "status": "ok",
            "num_input_frames": len(selected),
            "num_reg_images": int(reconstruction.num_reg_images()),
            "num_points3D": int(reconstruction.num_points3D()),
            "reprojection_error": round(reprojection_error, 4),
            "plane_inlier_ratio": round(float(plane["inlier_ratio"]), 4),
            "coverage": round(coverage, 4),
            "manhattan_ratio": round(manhattan_ratio, 4),
            "rotation_deg": round(rotation_deg, 4),
            "sharpness": round(sharpness, 4),
            "score": round(score, 4),
            "output_width": int(render["width"]),
            "output_height": int(render["height"]),
        }
        attempts_log.append(attempt_payload)

        candidate = {
            "score": score,
            "reprojection_error": reprojection_error,
            "sharpness": sharpness,
            "coverage": coverage,
            "manhattan_ratio": manhattan_ratio,
            "rotation_deg": rotation_deg,
            "reg_ratio": reg_ratio,
            "reprojection_score": reproj_score,
            "reconstruction": reconstruction,
            "ortho": ortho,
            "plane": plane,
            "render": render,
            "variant_name": attempt_name,
            "num_input_frames": len(selected),
            "num_reg_images": int(reconstruction.num_reg_images()),
            "num_points3D": int(reconstruction.num_points3D()),
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

        if candidate["score"] >= early_stop_score:
            break

    if best is None:
        quality_payload = {
            "backend": "pycolmap",
            "status": "failed",
            "rectification_score": 0.0,
            "attempts": attempts_log,
        }
        write_json(out_dir / "quality.json", quality_payload)
        raise RuntimeError("PyCOLMAP geometry could not produce a valid orthomosaic")

    out_ortho = out_dir / "facade_ortho.png"
    cv2.imwrite(str(out_ortho), best["ortho"])

    plane_payload = {
        "method": "pycolmap_sparse_plane_projection",
        "normal": [float(x) for x in best["plane"]["normal"]],
        "point": [float(x) for x in best["plane"]["point"]],
        "u_axis": [float(x) for x in best["plane"]["u_axis"]],
        "v_axis": [float(x) for x in best["plane"]["v_axis"]],
        "inlier_ratio": round(float(best["plane"]["inlier_ratio"]), 4),
        "confidence": round(float(best["score"]), 4),
        "selected_variant": best["variant_name"],
        "num_reg_images": int(best["num_reg_images"]),
        "num_points3D": int(best["num_points3D"]),
    }
    write_json(out_dir / "plane.json", plane_payload)

    quality_payload = {
        "backend": "pycolmap",
        "status": "ok",
        "rectification_score": round(float(best["score"]), 4),
        "sharpness": round(float(best["sharpness"]), 4),
        "coverage": round(float(best["coverage"]), 4),
        "manhattan_ratio": round(float(best["manhattan_ratio"]), 4),
        "rotation_deg": round(float(best["rotation_deg"]), 4),
        "reprojection_error": round(float(best["reprojection_error"]), 4),
        "reprojection_score": round(float(best["reprojection_score"]), 4),
        "registration_ratio": round(float(best["reg_ratio"]), 4),
        "selected_variant": best["variant_name"],
        "num_input_frames": int(best["num_input_frames"]),
        "num_reg_images": int(best["num_reg_images"]),
        "num_points3D": int(best["num_points3D"]),
        "output_width": int(best["render"]["width"]),
        "output_height": int(best["render"]["height"]),
        "num_attempts": len(attempts_log),
        "attempts": attempts_log,
    }
    write_json(out_dir / "quality.json", quality_payload)

    if float(best["score"]) < min_rectification_score:
        raise RuntimeError(
            "PyCOLMAP geometry failed quality gate: "
            f"rectification_score={best['score']:.3f} < {min_rectification_score:.3f}"
        )

    return {
        "facade_ortho_path": str(out_ortho),
        "plane_path": str(out_dir / "plane.json"),
        "quality_path": str(out_dir / "quality.json"),
        "width": int(best["render"]["width"]),
        "height": int(best["render"]["height"]),
        "rectification_score": float(best["score"]),
    }


def _run_geometry_homography(
    frame_paths: list[Path],
    out_dir: Path,
    cfg: dict[str, Any],
    logger,
    min_rectification_score: float,
) -> dict[str, Any]:
    sampling_top_k = int(cfg.get("sampling_top_k", 12))
    canny_variants = list(cfg.get("canny_variants", [[50, 150], [30, 120], [80, 200]]))
    angle_tolerance_deg = float(cfg.get("line_angle_tolerance_deg", 10.0))

    scored_frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        sharpness = _variance_of_laplacian(frame)
        scored_frames.append((frame_path, sharpness))

    if not scored_frames:
        raise RuntimeError("Geometry stage could not read any frames")

    scored_frames.sort(key=lambda x: x[1], reverse=True)
    selected_frames = [path for path, _ in scored_frames[:sampling_top_k]]
    logger.info(
        "Geometry stage (homography): evaluating %d frames with %d Canny variants",
        len(selected_frames),
        len(canny_variants),
    )

    best_attempt: RectificationAttempt | None = None
    attempts_log: list[dict[str, Any]] = []

    for frame_path in selected_frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        for canny in canny_variants:
            low, high = int(canny[0]), int(canny[1])
            attempt = _rectify_frame(frame_path, frame, low, high, angle_tolerance_deg)
            attempts_log.append(
                {
                    "frame": frame_path.name,
                    "canny": [low, high],
                    "score": round(attempt.score, 4),
                    "sharpness": round(attempt.sharpness, 4),
                    "coverage": round(attempt.coverage, 4),
                    "manhattan_ratio": round(attempt.manhattan_ratio, 4),
                    "had_quad": attempt.had_quad,
                }
            )
            if best_attempt is None or attempt.score > best_attempt.score:
                best_attempt = attempt

    if best_attempt is None:
        raise RuntimeError("Homography geometry failed to produce any rectification attempt")

    out_ortho = out_dir / "facade_ortho.png"
    cv2.imwrite(str(out_ortho), best_attempt.ortho)

    plane_payload = {
        "method": "homography_baseline",
        "normal": [0.0, 0.0, 1.0],
        "point": [0.0, 0.0, 0.0],
        "confidence": round(float(best_attempt.score), 4),
        "selected_frame": best_attempt.frame_path.name,
    }
    write_json(out_dir / "plane.json", plane_payload)

    quality_payload = {
        "backend": "homography_baseline",
        "rectification_score": round(float(best_attempt.score), 4),
        "sharpness": round(float(best_attempt.sharpness), 4),
        "coverage": round(float(best_attempt.coverage), 4),
        "manhattan_ratio": round(float(best_attempt.manhattan_ratio), 4),
        "selected_frame": best_attempt.frame_path.name,
        "selected_canny": [best_attempt.canny_low, best_attempt.canny_high],
        "num_attempts": len(attempts_log),
        "attempts": attempts_log[:200],
    }
    write_json(out_dir / "quality.json", quality_payload)

    if best_attempt.score < min_rectification_score:
        raise RuntimeError(
            "Homography geometry failed quality gate: "
            f"rectification_score={best_attempt.score:.3f} < {min_rectification_score:.3f}"
        )

    h, w = best_attempt.ortho.shape[:2]
    return {
        "facade_ortho_path": str(out_ortho),
        "plane_path": str(out_dir / "plane.json"),
        "quality_path": str(out_dir / "quality.json"),
        "width": int(w),
        "height": int(h),
        "rectification_score": float(best_attempt.score),
    }


def _collect_points3d(reconstruction) -> np.ndarray:
    points = [np.asarray(reconstruction.point3D(pid).xyz, dtype=np.float64) for pid in reconstruction.point3D_ids()]
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.stack(points, axis=0)


def _fit_plane_ransac(
    points: np.ndarray,
    dist_ratio: float = 0.01,
    max_iters: int = 500,
) -> dict[str, Any] | None:
    if points.shape[0] < 50:
        return None

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    if diag <= 1e-9:
        return None
    threshold = max(diag * max(dist_ratio, 1e-4), 1e-5)

    rng = np.random.default_rng(42)
    best_inliers: np.ndarray | None = None
    best_count = 0

    for _ in range(max_iters):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p1))
        dist = np.abs(points @ normal + d)
        inliers = dist < threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 50:
        return None

    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    centered = inlier_points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)

    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, up_ref))) > 0.95:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u_axis = np.cross(up_ref, normal)
    u_axis = u_axis / max(float(np.linalg.norm(u_axis)), 1e-9)
    v_axis = np.cross(normal, u_axis)
    v_axis = v_axis / max(float(np.linalg.norm(v_axis)), 1e-9)

    uv = np.column_stack((centered @ u_axis, centered @ v_axis))
    umin, umax = np.percentile(uv[:, 0], [1, 99])
    vmin, vmax = np.percentile(uv[:, 1], [1, 99])
    if float(umax - umin) <= 1e-6 or float(vmax - vmin) <= 1e-6:
        return None

    return {
        "normal": normal.astype(np.float64),
        "point": centroid.astype(np.float64),
        "u_axis": u_axis.astype(np.float64),
        "v_axis": v_axis.astype(np.float64),
        "umin": float(umin),
        "umax": float(umax),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "inlier_ratio": float(best_count / points.shape[0]),
    }


def _render_orthomosaic(
    reconstruction,
    frames_dir: Path,
    plane: dict[str, Any],
    max_output_size: int,
    min_output_size: int,
) -> dict[str, Any] | None:
    normal = np.asarray(plane["normal"], dtype=np.float64)
    point = np.asarray(plane["point"], dtype=np.float64)
    u_axis = np.asarray(plane["u_axis"], dtype=np.float64)
    v_axis = np.asarray(plane["v_axis"], dtype=np.float64)
    umin, umax = float(plane["umin"]), float(plane["umax"])
    vmin, vmax = float(plane["vmin"]), float(plane["vmax"])

    scale_samples = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        p0 = image.project_point(point)
        pu = image.project_point(point + u_axis)
        pv = image.project_point(point + v_axis)
        if p0 is None or pu is None or pv is None:
            continue
        du = float(np.linalg.norm(np.asarray(pu) - np.asarray(p0)))
        dv = float(np.linalg.norm(np.asarray(pv) - np.asarray(p0)))
        if du > 1e-3 and dv > 1e-3:
            scale_samples.append((du + dv) * 0.5)
    scale = float(np.median(scale_samples)) if scale_samples else 300.0

    width = max(int(round((umax - umin) * scale)), 64)
    height = max(int(round((vmax - vmin) * scale)), 64)

    max_dim = max(width, height)
    min_dim = min(width, height)
    if max_dim > max_output_size:
        resize = max_output_size / max_dim
        scale *= resize
        width = max(int(round(width * resize)), 64)
        height = max(int(round(height * resize)), 64)
    if min_dim < min_output_size:
        resize = min_output_size / max(min_dim, 1)
        scale *= resize
        width = max(int(round(width * resize)), 64)
        height = max(int(round(height * resize)), 64)

    if width < 64 or height < 64:
        return None

    dst_corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    world_corners = np.array(
        [
            point + u_axis * umin + v_axis * vmin,
            point + u_axis * umax + v_axis * vmin,
            point + u_axis * umax + v_axis * vmax,
            point + u_axis * umin + v_axis * vmax,
        ],
        dtype=np.float64,
    )

    accum = np.zeros((height, width, 3), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)

    sharpness_by_name: dict[str, float] = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        frame = cv2.imread(str(frames_dir / image.name))
        if frame is None:
            continue
        sharpness_by_name[image.name] = _variance_of_laplacian(frame)

    sharpness_max = max(sharpness_by_name.values()) if sharpness_by_name else 1.0

    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        source = cv2.imread(str(frames_dir / image.name))
        if source is None:
            continue

        src_points: list[list[float]] = []
        valid_projection = True
        for world_pt in world_corners:
            proj = image.project_point(world_pt)
            if proj is None:
                valid_projection = False
                break
            src_points.append([float(proj[0]), float(proj[1])])
        if not valid_projection:
            continue

        src = np.float32(src_points)
        H = cv2.getPerspectiveTransform(dst_corners, src)
        warped = cv2.warpPerspective(source, H, (width, height), flags=cv2.INTER_LINEAR)
        mask_src = np.full((source.shape[0], source.shape[1]), 255, dtype=np.uint8)
        valid = cv2.warpPerspective(mask_src, H, (width, height), flags=cv2.INTER_NEAREST) > 0
        if float(valid.mean()) < 0.01:
            continue

        view_dir = np.asarray(image.viewing_direction(), dtype=np.float64)
        view_norm = max(float(np.linalg.norm(view_dir)), 1e-9)
        angle_weight = abs(float(np.dot(view_dir / view_norm, normal)))
        sharp = sharpness_by_name.get(image.name, 0.0)
        sharp_weight = min(float(sharp / max(sharpness_max, 1e-6)), 1.0)
        weight = max(0.05, 0.6 * angle_weight + 0.4 * sharp_weight)

        accum[valid] += warped[valid].astype(np.float64) * weight
        weights[valid] += weight

    valid_output = weights > 1e-6
    if not np.any(valid_output):
        return None

    ortho = np.full((height, width, 3), 255, dtype=np.uint8)
    ortho[valid_output] = np.clip(
        (accum[valid_output] / weights[valid_output, None]),
        0.0,
        255.0,
    ).astype(np.uint8)
    coverage = float(valid_output.mean())

    return {
        "ortho": ortho,
        "width": int(width),
        "height": int(height),
        "coverage": coverage,
    }


def _rectify_frame(
    frame_path: Path,
    frame: np.ndarray,
    canny_low: int,
    canny_high: int,
    angle_tolerance_deg: float,
) -> RectificationAttempt:
    quad, coverage = _detect_main_quad(frame, canny_low, canny_high)
    had_quad = quad is not None

    if quad is not None:
        rect = _order_points(quad)
        warped = _warp_perspective(frame, rect)
    else:
        warped = frame.copy()
        coverage = 0.3

    sharpness = _variance_of_laplacian(warped)
    manhattan_ratio = _manhattan_ratio(warped, angle_tolerance_deg)
    sharpness_norm = min(sharpness / 1200.0, 1.0)
    score = 0.45 * sharpness_norm + 0.35 * manhattan_ratio + 0.20 * float(np.clip(coverage, 0.0, 1.0))
    if not had_quad:
        score -= 0.12
    score = float(np.clip(score, 0.0, 1.0))

    return RectificationAttempt(
        frame_path=frame_path,
        canny_low=canny_low,
        canny_high=canny_high,
        score=score,
        sharpness=float(sharpness),
        coverage=float(coverage),
        manhattan_ratio=float(manhattan_ratio),
        had_quad=had_quad,
        ortho=warped,
    )


def _detect_main_quad(frame: np.ndarray, canny_low: int, canny_high: int) -> tuple[np.ndarray | None, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    image_area = float(h * w)
    best_quad = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.08 * image_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2).astype(np.float32)
            best_area = area

    if best_quad is None:
        return None, 0.0

    coverage = best_area / max(image_area, 1.0)
    return best_quad, float(np.clip(coverage, 0.0, 1.0))


def _order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def _warp_perspective(frame: np.ndarray, rect: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    width = max(int(max(width_a, width_b)), 64)
    height = max(int(max(height_a, height_b)), 64)

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(frame, matrix, (width, height))


def _variance_of_laplacian(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _manhattan_ratio(image_bgr: np.ndarray, tolerance_deg: float) -> float:
    angles = _extract_line_angles(image_bgr)
    if angles.size == 0:
        return 0.0

    abs_angle = np.abs(angles)
    dist = np.minimum(abs_angle, np.abs(abs_angle - 90.0))
    return float(np.mean(dist <= tolerance_deg))


def _align_to_manhattan(image_bgr: np.ndarray, tolerance_deg: float) -> tuple[np.ndarray, float, float]:
    angles = _extract_line_angles(image_bgr)
    if angles.size == 0:
        return image_bgr, 0.0, 0.0

    best_theta = 0.0
    best_score = -1.0
    best_mean_dist = 1e9

    for theta in np.linspace(-35.0, 35.0, num=281):
        adjusted = _normalize_angles(angles - theta)
        abs_angle = np.abs(adjusted)
        dist = np.minimum(abs_angle, np.abs(abs_angle - 90.0))
        score = float(np.mean(dist <= tolerance_deg))
        mean_dist = float(np.mean(dist))
        if score > best_score or (abs(score - best_score) < 1e-9 and mean_dist < best_mean_dist):
            best_score = score
            best_mean_dist = mean_dist
            best_theta = float(theta)

    if abs(best_theta) < 0.25:
        manhattan = _manhattan_ratio(image_bgr, tolerance_deg)
        return image_bgr, 0.0, manhattan

    h, w = image_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_m = cv2.getRotationMatrix2D(center, best_theta, 1.0)
    rotated = cv2.warpAffine(
        image_bgr,
        rot_m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    manhattan = _manhattan_ratio(rotated, tolerance_deg)
    return rotated, best_theta, manhattan


def _extract_line_angles(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=max(min(gray.shape) // 6, 20),
        maxLineGap=8,
    )
    if lines is None or len(lines) == 0:
        return np.empty((0,), dtype=np.float64)

    angles: list[float] = []
    for item in lines:
        x1, y1, x2, y2 = item[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(float(angle))
    return _normalize_angles(np.asarray(angles, dtype=np.float64))


def _normalize_angles(angles: np.ndarray) -> np.ndarray:
    norm = (angles + 90.0) % 180.0 - 90.0
    return norm


def _read_json_safe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))
