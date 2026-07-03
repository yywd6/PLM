"""Build the source-by-target point-AUROC matrix for one-rest runs."""

import argparse
import csv
from pathlib import Path

from evaluation_metrics import finite_mean
from one_rest_protocol import all_categories, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize one-rest point AUROC.")
    parser.add_argument("--output_root", default="outputs/one_rest")
    parser.add_argument("--dataset_name", default="Real3D")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    categories = all_categories(args.dataset_name)
    rows = []
    json_rows = {}
    for source in categories:
        metrics_path = output_root / source / "per_category_metrics.json"
        metrics = {}
        if metrics_path.is_file():
            import json

            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle).get("metrics", {})
        values = {
            target: (
                metrics.get(target, {}).get("point_auroc")
                if target != source
                else None
            )
            for target in categories
        }
        mean = finite_mean(values.values())
        rows.append({"train_category": source, **values, "mean": mean})
        json_rows[source] = {**values, "mean": mean}

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_category", *categories, "mean"])
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_root / "summary.json", {
        "dataset": args.dataset_name,
        "metric": "point_auroc",
        "rows": json_rows,
    })
    print(f"Summary: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
