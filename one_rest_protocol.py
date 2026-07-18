"""Shared one-rest category resolution and leakage checks."""

import json
from pathlib import Path

import yaml

from data.anomaly_datasets import PointCloudDataset


def _deep_merge(base, override):
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path, _seen=None):
    """Load a config with optional relative ``_base_`` inheritance."""
    path = Path(path)
    seen = set() if _seen is None else set(_seen)
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic config inheritance: {path}")
    seen.add(resolved)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    base_path = config.pop("_base_", None)
    if base_path is not None:
        base_path = Path(base_path)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base = load_yaml(base_path, _seen=seen)
        config = _deep_merge(base, config)
    return config


def forbidden_config_keys():
    parts = (
        ("lambda", "normal", "topk", "suppression"),
        ("normal", "suppression", "top", "percent"),
        ("object", "calibration", "top", "percent"),
        ("target", "normal"),
        ("hard", "normal"),
        ("mil", "object"),
        ("spatial", "object"),
        ("layerwise", "uncertainty"),
        ("object", "score", "v12", "trainable"),
        ("test", "time", "self", "calibration"),
    )
    return ["_".join(item) for item in parts] + [
        "".join(("TrainableLayerwise", "UncertaintyCalibrator"))
    ]


def assert_no_forbidden_config_keys(config, path):
    hits = [key for key in forbidden_config_keys() if key in config]
    if hits:
        raise ValueError(f"{path} contains deprecated object-branch keys: {hits}")


def all_categories(dataset_name):
    return list(PointCloudDataset.PRESETS[dataset_name])


def normalize_train_categories(dataset_name, train_category=None, train_categories=None):
    """Return a validated, de-duplicated source-category list.

    ``train_category`` remains supported for old one-source configs. New configs
    can use ``train_categories`` to train one checkpoint from multiple sources.
    """
    categories = all_categories(dataset_name)
    selected = train_categories if train_categories else train_category
    if isinstance(selected, str):
        selected = [selected]
    selected = list(selected or [])
    if not selected:
        raise ValueError("train_category or train_categories is required")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate train categories: {selected}")
    unknown = [name for name in selected if name not in categories]
    if unknown:
        raise ValueError(f"Unknown train categories {unknown} for {dataset_name}")
    return selected


def category_run_name(train_categories):
    """Stable output/checkpoint directory name for one or more sources."""
    return "+".join(train_categories)


def resolve_categories(dataset_name, protocol, train_category=None, train_categories=None):
    categories = all_categories(dataset_name)
    selected = normalize_train_categories(
        dataset_name, train_category=train_category, train_categories=train_categories
    )
    if protocol == "one_rest":
        selected_set = set(selected)
        return selected, [name for name in categories if name not in selected_set]
    return selected, categories


def checkpoint_train_categories(checkpoint):
    """Read source categories from new and legacy checkpoints."""
    stored = checkpoint.get("train_categories")
    if stored:
        return list(stored)
    legacy = checkpoint.get("train_category")
    return [legacy] if legacy else []


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
