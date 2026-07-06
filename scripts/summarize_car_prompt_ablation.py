#!/usr/bin/env python3
"""Summarize the three car-source prompt-template ablations."""

import argparse
import csv
import json
from pathlib import Path


VARIANTS = (
    ("fixed_generic", "point cloud patch"),
    ("category_only", "{category}"),
    ("category_semantic", "a point cloud patch of {article} {category}"),
)
METRICS = ("object_auroc", "object_ap", "point_auroc", "point_ap", "point_pro")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base", default="outputs/car_prompt_ablation")
    args = parser.parse_args()
    output_base = Path(args.output_base)

    rows = []
    for variant, template in VARIANTS:
        metrics_path = output_base / variant / "car" / "mean_metrics.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing ablation result: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "variant": variant,
            "template": template,
            **{name: metrics[name] for name in METRICS},
        }
        row["object_point_auroc_sum"] = row["object_auroc"] + row["point_auroc"]
        rows.append(row)

    rows.sort(key=lambda row: row["object_point_auroc_sum"], reverse=True)
    output_base.mkdir(parents=True, exist_ok=True)
    csv_path = output_base / "prompt_ablation_summary.csv"
    json_path = output_base / "prompt_ablation_summary.json"
    fieldnames = ("variant", "template", *METRICS, "object_point_auroc_sum")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Prompt ablation ranking by O-AUROC + P-AUROC:")
    for rank, row in enumerate(rows, 1):
        print(
            f"{rank}. {row['variant']}: "
            f"O-AUROC={100 * row['object_auroc']:.2f}, "
            f"P-AUROC={100 * row['point_auroc']:.2f}, "
            f"sum={100 * row['object_point_auroc_sum']:.2f}"
        )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
