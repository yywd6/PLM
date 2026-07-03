"""Evaluate the frozen ULIP-2 zero-shot object anomaly baseline."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline import anomaly_probability, binary_metrics, finite_mean
from data.anomaly_datasets import PointCloudDataset
from models.ulip2_encoder import ULIP2Encoder


DEFAULT_CONFIG = "configs/ulip_zero_shot_baseline.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def build_parser():
    parser = argparse.ArgumentParser(description="Frozen ULIP-2 zero-shot object anomaly baseline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data_root")
    parser.add_argument("--model_path")
    parser.add_argument("--dataset_name", choices=sorted(PointCloudDataset.PRESETS), default="Real3D")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--return_layers", type=int, nargs="+", default=[2, 5, 8, 11])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--normal_templates", nargs="+")
    parser.add_argument("--anomaly_templates", nargs="+")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output_dir", default="outputs/ulip_zero_shot_baseline")
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)
    parser = build_parser()
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown baseline config keys: {unknown_keys}")
    parser.set_defaults(**config)
    args = parser.parse_args()
    required = ("data_root", "model_path", "normal_templates", "anomaly_templates")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        parser.error(f"missing required configuration: {', '.join(missing)}")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    return args


def json_number(value):
    return float(value) if np.isfinite(value) else None


@torch.inference_mode()
def evaluate(args):
    classes = args.classes or PointCloudDataset.PRESETS[args.dataset_name]
    unknown = sorted(set(classes) - set(PointCloudDataset.PRESETS[args.dataset_name]))
    if unknown:
        raise ValueError(f"Unknown classes for {args.dataset_name}: {unknown}")
    dataset = PointCloudDataset(
        root_dir=args.data_root, split=args.split, classes=classes, dataset_name=args.dataset_name
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No samples found under {args.data_root} for {args.dataset_name}/{args.split}"
        )
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        pin_memory=device == "cuda", num_workers=args.num_workers,
    )
    encoder = ULIP2Encoder(
        args.model_path,
        device=device,
        num_points=args.num_points,
        return_layers=tuple(args.return_layers),
    )
    encoder.model.eval()
    normal_embedding = encoder.encode_text_templates(args.normal_templates)
    anomaly_embedding = encoder.encode_text_templates(args.anomaly_templates)

    labels_by_class = defaultdict(list)
    scores_by_class = defaultdict(list)
    class_names = PointCloudDataset.PRESETS[args.dataset_name]
    for batch in tqdm(loader, desc="Evaluate", dynamic_ncols=True):
        points = batch["points"].to(device, non_blocking=True)
        embeddings = encoder.encode_pointcloud(points)["concat"]
        scores = anomaly_probability(
            embeddings, normal_embedding, anomaly_embedding, args.temperature
        ).cpu().tolist()
        labels = (batch["labels"].reshape(batch["labels"].shape[0], -1) > 0).any(dim=1)
        for category, label, score in zip(batch["category"], labels, scores):
            class_name = class_names[int(category)]
            labels_by_class[class_name].append(int(label))
            scores_by_class[class_name].append(float(score))

    per_class = {}
    for class_name in classes:
        metrics = binary_metrics(labels_by_class[class_name], scores_by_class[class_name])
        per_class[class_name] = {
            "samples": len(labels_by_class[class_name]),
            **{name: json_number(value) for name, value in metrics.items()},
        }
    macro = {}
    for name in ("auroc", "ap", "f1_max"):
        values = [metrics[name] for metrics in per_class.values() if metrics[name] is not None]
        macro[name] = json_number(finite_mean(values))
    return {"dataset": args.dataset_name, "split": args.split, "macro": macro, "per_class": per_class}


def save_results(args, results):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    with (output_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("class", "samples", "object_auroc", "object_ap", "object_f1_max"))
        for class_name, metrics in results["per_class"].items():
            writer.writerow((class_name, metrics["samples"], metrics["auroc"], metrics["ap"], metrics["f1_max"]))


def main():
    args = parse_args()
    results = evaluate(args)
    save_results(args, results)
    print(json.dumps(results["macro"], indent=2))
    print(f"Results: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    import sys

    if "--protocol" in sys.argv:
        from test_standard_aupro import main as protocol_main

        protocol_main()
    else:
        main()
