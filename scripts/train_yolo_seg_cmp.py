from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight YOLO-seg model on CMP (windows/doors)")
    parser.add_argument("--data", default="data/cmp_yolo_seg/dataset.yaml", help="YOLO dataset yaml path")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--device", default="cpu", help="cpu/cuda")
    parser.add_argument("--project", default="runs/yolo_cmp", help="Ultralytics project dir")
    parser.add_argument("--name", default="seg_small", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--patience", type=int, default=20, help="Early stop patience")
    parser.add_argument("--resume", action="store_true", help="Resume from existing run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ultralytics is unavailable: {exc}") from exc

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    model = YOLO(args.model)
    result = model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        device=args.device,
        project=args.project,
        name=args.name,
        seed=int(args.seed),
        patience=int(args.patience),
        close_mosaic=0,
        optimizer="auto",
        plots=True,
        verbose=True,
        cache=False,
        resume=bool(args.resume),
    )

    save_dir = Path(result.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    metrics = {}
    try:
        metrics = result.results_dict
    except Exception:  # noqa: BLE001
        metrics = {}

    payload = {
        "save_dir": str(save_dir.resolve()),
        "best_pt": str(best_pt.resolve()) if best_pt.exists() else "",
        "last_pt": str(last_pt.resolve()) if last_pt.exists() else "",
        "metrics": metrics,
    }
    summary_path = save_dir / "train_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

