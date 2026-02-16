from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CLASS_TO_ID = {"window": 0, "door": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CMP Facade data in YOLO-seg format")
    parser.add_argument("--dataset-dir", default="data/cmp_facade_base/base", help="CMP base folder with .jpg/.xml")
    parser.add_argument("--outdir", default="data/cmp_yolo_seg", help="YOLO dataset output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0=all)")
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include images without windows/doors",
    )
    parser.add_argument(
        "--task",
        choices=["seg", "det"],
        default="seg",
        help="Output label type: segmentation polygons or detection boxes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    outdir = Path(args.outdir)
    image_paths = sorted(dataset_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No .jpg files found in {dataset_dir}")

    records: list[dict[str, Any]] = []
    for image_path in image_paths:
        xml_path = image_path.with_suffix(".xml")
        if not xml_path.exists():
            continue
        objects = parse_cmp_xml(xml_path)
        if not args.include_empty and not objects:
            continue
        records.append({"image_path": image_path, "objects": objects})

    if not records:
        raise RuntimeError("No samples were collected after filtering")

    if args.max_images > 0:
        records = records[: args.max_images]

    rng = random.Random(args.seed)
    rng.shuffle(records)

    val_count = max(1, int(round(len(records) * args.val_ratio)))
    val_records = records[:val_count]
    train_records = records[val_count:]
    if not train_records:
        train_records, val_records = records[:-1], records[-1:]

    if outdir.exists():
        shutil.rmtree(outdir)
    for split in ("train", "val"):
        (outdir / "images" / split).mkdir(parents=True, exist_ok=True)
        (outdir / "labels" / split).mkdir(parents=True, exist_ok=True)

    write_split(train_records, outdir, "train", task=args.task)
    write_split(val_records, outdir, "val", task=args.task)

    yaml_path = outdir / "dataset.yaml"
    outdir_posix = outdir.resolve().as_posix()
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {outdir_posix}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: window",
                "  1: door",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "outdir": str(outdir.resolve()),
        "task": args.task,
        "num_total": len(records),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "num_train_objects": sum(len(item["objects"]) for item in train_records),
        "num_val_objects": sum(len(item["objects"]) for item in val_records),
        "class_to_id": CLASS_TO_ID,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote dataset: {outdir}")
    print(f"Wrote yaml: {yaml_path}")
    print(json.dumps(summary, indent=2))


def write_split(records: list[dict[str, Any]], outdir: Path, split: str, task: str) -> None:
    for item in records:
        image_path: Path = item["image_path"]
        objects: list[dict[str, Any]] = item["objects"]
        out_image = outdir / "images" / split / image_path.name
        out_label = outdir / "labels" / split / f"{image_path.stem}.txt"

        shutil.copyfile(image_path, out_image)

        lines = []
        for obj in objects:
            cls_id = CLASS_TO_ID[obj["label"]]
            x1, y1, x2, y2 = obj["bbox_norm"]
            if task == "det":
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                bw = x2 - x1
                bh = y2 - y1
                line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            else:
                poly = [x1, y1, x2, y1, x2, y2, x1, y2]
                line = f"{cls_id} " + " ".join(f"{v:.6f}" for v in poly)
            lines.append(line)
        out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_cmp_xml(xml_path: Path) -> list[dict[str, Any]]:
    text = xml_path.read_text(encoding="utf-8").strip()
    wrapped = f"<root>{text}</root>"
    root = ET.fromstring(wrapped)

    objects: list[dict[str, Any]] = []
    for obj in root.findall("object"):
        label_name = (obj.findtext("labelname") or "").strip().lower()
        if label_name not in CLASS_TO_ID:
            continue

        x_nodes = obj.findall("points/x")
        y_nodes = obj.findall("points/y")
        if len(x_nodes) < 2 or len(y_nodes) < 2:
            continue

        x_vals = sorted(float(node.text.strip()) for node in x_nodes if node.text)
        y_vals = sorted(float(node.text.strip()) for node in y_nodes if node.text)
        if len(x_vals) < 2 or len(y_vals) < 2:
            continue

        x1 = float(max(0.0, min(1.0, x_vals[0])))
        x2 = float(max(0.0, min(1.0, x_vals[-1])))
        y1 = float(max(0.0, min(1.0, y_vals[0])))
        y2 = float(max(0.0, min(1.0, y_vals[-1])))
        if x2 <= x1 or y2 <= y1:
            continue

        objects.append({"label": label_name, "bbox_norm": [x1, y1, x2, y2]})
    return objects


if __name__ == "__main__":
    main()
