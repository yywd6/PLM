#!/usr/bin/env python3
"""Summarize selected two-rest runs and average the three runs per dataset."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


METRIC_NAMES = (
    "object_auroc",
    "object_ap",
    "original_object_auroc",
    "original_object_ap",
    "point_auroc",
    "point_ap",
    "point_pro",
)
def finite_or_none(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def finite_mean(values):
    valid = [value for value in (finite_or_none(item) for item in values) if value is not None]
    return sum(valid) / len(valid) if valid else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average mean_metrics.json across selected two-rest runs."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=3,
        help="Required number of completed combinations per dataset; 0 disables the check.",
    )
    return parser.parse_args()


def load_runs(output_root):
    grouped = defaultdict(list)
    for metrics_path in sorted(output_root.glob("*/*/mean_metrics.json")):
        dataset_name = metrics_path.parent.parent.name
        pair_name = metrics_path.parent.name
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        grouped[dataset_name].append({
            "pair": pair_name,
            "metrics_path": str(metrics_path),
            "test_categories": payload.get("test_categories", []),
            "metrics": {
                name: finite_or_none(payload.get(name)) for name in METRIC_NAMES
            },
        })
    return grouped


def build_summary(grouped, expected_runs):
    if not grouped:
        raise FileNotFoundError("No */*/mean_metrics.json files were found")
    datasets = {}
    for dataset_name, runs in sorted(grouped.items()):
        if expected_runs > 0 and len(runs) != expected_runs:
            raise RuntimeError(
                f"{dataset_name}: expected {expected_runs} completed runs, found {len(runs)}"
            )
        datasets[dataset_name] = {
            "run_count": len(runs),
            "averaging": "arithmetic mean of the run-level macro metrics",
            "runs": runs,
            "mean_metrics": {
                name: finite_mean(run["metrics"][name] for run in runs)
                for name in METRIC_NAMES
            },
        }
    return {"expected_runs_per_dataset": expected_runs, "datasets": datasets}


def write_outputs(output_root, summary):
    json_path = output_root / "selected_two_rest_summary.json"
    csv_path = output_root / "selected_two_rest_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset", "row_type", "pair", *METRIC_NAMES
            ),
        )
        writer.writeheader()
        for dataset_name, dataset in summary["datasets"].items():
            for run in dataset["runs"]:
                writer.writerow({
                    "dataset": dataset_name,
                    "row_type": "combination",
                    "pair": run["pair"],
                    **run["metrics"],
                })
            writer.writerow({
                "dataset": dataset_name,
                "row_type": "dataset_mean",
                "pair": "MEAN",
                **dataset["mean_metrics"],
            })
    return json_path, csv_path


def format_metric(value):
    return "NA" if value is None else f"{value:.6f}"


def main():
    args = parse_args()
    grouped = load_runs(args.output_root)
    summary = build_summary(grouped, args.expected_runs)
    json_path, csv_path = write_outputs(args.output_root, summary)
    for dataset_name, dataset in summary["datasets"].items():
        print(f"{dataset_name}: {dataset['run_count']} completed combinations")
        for run in dataset["runs"]:
            metrics = run["metrics"]
            print(
                f"  {run['pair']}: object_auroc={format_metric(metrics['object_auroc'])} "
                f"point_auroc={format_metric(metrics['point_auroc'])} "
                f"point_pro={format_metric(metrics['point_pro'])}"
            )
        means = dataset["mean_metrics"]
        print(
            f"  MEAN: object_auroc={format_metric(means['object_auroc'])} "
            f"point_auroc={format_metric(means['point_auroc'])} "
            f"point_pro={format_metric(means['point_pro'])}"
        )
    print(f"JSON summary: {json_path}")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
