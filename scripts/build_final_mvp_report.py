from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final A->E MVP summary report from stage comparison files")
    parser.add_argument("--report-dir", default="reports", help="Directory with comparison reports")
    parser.add_argument("--tag", default="mvp_final_summary", help="Output file prefix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    stage_a_seq = read_json(report_dir / "geometry_compare_seq.json")
    stage_a_single = read_json(report_dir / "geometry_compare_single.json")
    stage_b = read_json(report_dir / "cmp_seg_backend_compare_30.json")
    stage_c = read_json(report_dir / "cmp_line_backend_compare_30.json")
    stage_d = read_json(report_dir / "cmp_vector_variant_compare_30.json")
    stage_e = read_json(report_dir / "cmp_export_variant_compare_30.json")

    a_seq_winner = pick_a_seq_winner(stage_a_seq)
    a_single_winner = pick_a_single_winner(stage_a_single)
    b_winner = stage_b.get("winner")
    c_winner = stage_c.get("winner")
    d_winner = stage_d.get("winner")
    e_winner = stage_e.get("winner")

    final = {
        "generated_on": str(date.today().isoformat()),
        "stage_winners": {
            "A_seq": a_seq_winner.get("backend", "n/a"),
            "A_single": a_single_winner.get("backend", "n/a"),
            "B_segmentation": b_winner,
            "C_lines": c_winner,
            "D_vectorize": d_winner,
            "E_export": e_winner,
        },
        "recommended": {
            "default_config": "configs/mvp_final.yaml",
            "quality_config": "configs/mvp_final_quality.yaml",
            "one_command_default": (
                "python scripts/run_mvp_final.py --input <video_or_images_dir> "
                "--outdir runs/run_final_0001 --profile default --device cpu --seed 42"
            ),
            "one_command_quality": (
                "python scripts/run_mvp_final.py --input <video_or_images_dir> "
                "--outdir runs/run_final_quality_0001 --profile quality --device cpu --seed 42"
            ),
        },
        "notes": [
            "Stage A sequential comparison: PyCOLMAP has higher rectification score but much slower runtime.",
            "Stage A single-image comparison: PyCOLMAP profile falls back to homography baseline.",
            "CMP 30-image evaluations across B-E have 2 geometry-gate failures (cmp_b0326, cmp_b0352).",
            "Default final profile prioritizes robustness/speed on mixed single-image and short-sequence inputs.",
        ],
        "evidence": {
            "A_seq": stage_a_seq,
            "A_single": stage_a_single,
            "B": stage_b,
            "C": stage_c,
            "D": stage_d,
            "E": stage_e,
        },
    }

    json_path = report_dir / f"{args.tag}.json"
    json_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    md_path = report_dir / f"{args.tag}.md"
    md_path.write_text(build_markdown(final, a_seq_winner, a_single_winner), encoding="utf-8")

    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required report file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def pick_a_seq_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row.get("rectification_score", 0.0)),
            float(row.get("manhattan_ratio", 0.0)),
            -float(row.get("geometry_time_sec", 0.0)),
        )

    return max(rows, key=key) if rows else {}


def pick_a_single_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row.get("rectification_score", 0.0)),
            float(row.get("manhattan_ratio", 0.0)),
            -float(row.get("geometry_time_sec", 0.0)),
        )

    return max(rows, key=key) if rows else {}


def build_markdown(final: dict[str, Any], a_seq_winner: dict[str, Any], a_single_winner: dict[str, Any]) -> str:
    winners = final["stage_winners"]
    recommended = final["recommended"]
    notes: list[str] = final["notes"]

    lines = [
        "# Final MVP Summary (A->E)",
        "",
        f"- generated_on: {final['generated_on']}",
        "",
        "## Stage Winners",
        f"- A (sequential/multi-frame): `{winners['A_seq']}`",
        f"- A (single-image): `{winners['A_single']}`",
        f"- B segmentation: `{winners['B_segmentation']}`",
        f"- C lines: `{winners['C_lines']}`",
        f"- D vectorization: `{winners['D_vectorize']}`",
        f"- E export: `{winners['E_export']}`",
        "",
        "## A Evidence",
        (
            f"- sequential winner metrics: rectification={float(a_seq_winner.get('rectification_score', 0.0)):.4f}, "
            f"manhattan={float(a_seq_winner.get('manhattan_ratio', 0.0)):.4f}, "
            f"geometry_sec={float(a_seq_winner.get('geometry_time_sec', 0.0)):.4f}, "
            f"backend={a_seq_winner.get('backend', 'n/a')}"
        ),
        (
            f"- single-image winner metrics: rectification={float(a_single_winner.get('rectification_score', 0.0)):.4f}, "
            f"manhattan={float(a_single_winner.get('manhattan_ratio', 0.0)):.4f}, "
            f"geometry_sec={float(a_single_winner.get('geometry_time_sec', 0.0)):.4f}, "
            f"backend={a_single_winner.get('backend', 'n/a')}"
        ),
        "",
        "## Final Configs",
        f"- default: `{recommended['default_config']}`",
        f"- quality: `{recommended['quality_config']}`",
        "",
        "## One-Command Run",
        "```bash",
        recommended["one_command_default"],
        recommended["one_command_quality"],
        "```",
        "",
        "## Notes",
    ]
    for note in notes:
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## Source Reports",
            "- `reports/geometry_compare_seq.json`",
            "- `reports/geometry_compare_single.json`",
            "- `reports/cmp_seg_backend_compare_30.json`",
            "- `reports/cmp_line_backend_compare_30.json`",
            "- `reports/cmp_vector_variant_compare_30.json`",
            "- `reports/cmp_export_variant_compare_30.json`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
