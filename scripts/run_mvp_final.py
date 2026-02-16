from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline


PROFILE_TO_CONFIG = {
    "default": ROOT / "configs/mvp_final.yaml",
    "quality": ROOT / "configs/mvp_final_quality.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final MVP facade pipeline with a predefined profile")
    parser.add_argument("--input", required=True, help="Path to input video, image, or image directory")
    parser.add_argument("--outdir", required=True, help="Directory for output artifacts")
    parser.add_argument(
        "--profile",
        default="default",
        choices=sorted(PROFILE_TO_CONFIG.keys()),
        help="Final profile: default (fast/robust) or quality (PyCOLMAP-first)",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional explicit config path; overrides --profile mapping",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Execution device hint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve() if args.config else PROFILE_TO_CONFIG[args.profile]
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    run_pipeline(
        input_path=args.input,
        outdir=args.outdir,
        config_path=config_path,
        device=args.device,
        seed=args.seed,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
