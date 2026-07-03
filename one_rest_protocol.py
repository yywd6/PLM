"""Shared one-rest category resolution and leakage checks."""

import json
from pathlib import Path

import yaml

from data.anomaly_datasets import PointCloudDataset


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def all_categories(dataset_name):
    return list(PointCloudDataset.PRESETS[dataset_name])


def resolve_categories(dataset_name, protocol, train_category):
    categories = all_categories(dataset_name)
    if train_category not in categories:
        raise ValueError(f"Unknown train_category '{train_category}' for {dataset_name}")
    if protocol == "one_rest":
        return [train_category], [name for name in categories if name != train_category]
    return [train_category], categories


def validate_one_rest_flags(args):
    if args.protocol != "one_rest":
        return
    required_true = (
        "exclude_train_category_from_test",
        "zero_shot_target",
        "save_per_category_metrics",
        "save_mean_metrics",
    )
    disabled = [name for name in required_true if not getattr(args, name)]
    if disabled:
        raise ValueError(f"one_rest requires true flags: {disabled}")
    if args.use_target_anomaly_for_training:
        raise ValueError("Target-category samples are forbidden during one-rest training")


def sample_category(path):
    return Path(path).parent.parent.name


def assert_dataset_categories(dataset, allowed, forbidden=()):
    allowed = set(allowed)
    forbidden = set(forbidden)
    observed = {sample_category(path) for path in dataset.samples}
    unexpected = observed - allowed
    leaked = observed & forbidden
    if unexpected or leaked:
        raise RuntimeError(
            f"Dataset category leakage: observed={sorted(observed)}, "
            f"unexpected={sorted(unexpected)}, forbidden={sorted(leaked)}"
        )
    return sorted(observed)


def write_yaml(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=True, allow_unicode=True)


def write_json(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, ensure_ascii=False)


def log_line(path, message):
    print(message)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")
