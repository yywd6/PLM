"""Reusable integrity checks for resumable training and evaluation artifacts."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
import yaml


REQUIRED_MEAN_METRICS = (
    "object_auroc",
    "object_ap",
    "point_auroc",
    "point_ap",
    "point_pro",
)


def _nonempty(path):
    path = Path(path)
    if not path.is_file():
        return False, f"missing file: {path}"
    if path.stat().st_size <= 0:
        return False, f"empty file: {path}"
    return True, None


def validate_json_file(path, required_keys=()):
    valid, error = _nonempty(path)
    if not valid:
        return False, error
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return False, f"invalid JSON {path}: {exc}"
    if not isinstance(payload, dict):
        return False, f"JSON root is not a mapping: {path}"
    missing = sorted(set(required_keys) - set(payload))
    if missing:
        return False, f"JSON missing keys {missing}: {path}"
    return True, payload


def validate_yaml_file(path, required_keys=()):
    valid, error = _nonempty(path)
    if not valid:
        return False, error
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except Exception as exc:
        return False, f"invalid YAML {path}: {exc}"
    if not isinstance(payload, dict):
        return False, f"YAML root is not a mapping: {path}"
    missing = sorted(set(required_keys) - set(payload))
    if missing:
        return False, f"YAML missing keys {missing}: {path}"
    return True, payload


def validate_checkpoint_file(path, required_keys=("adapter",)):
    valid, error = _nonempty(path)
    if not valid:
        return False, error
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"unreadable checkpoint {path}: {exc}"
    if not isinstance(payload, dict):
        return False, f"checkpoint root is not a mapping: {path}"
    missing = sorted(set(required_keys) - set(payload))
    if missing:
        return False, f"checkpoint missing keys {missing}: {path}"
    return True, payload


def validate_ddf3d_checkpoint_payload(payload):
    """Validate explicit calibration/router components in a DDF-3D checkpoint."""
    metadata = payload.get("ddf3d")
    if not isinstance(metadata, dict) or metadata.get("enabled") is not True:
        return False, "DDF-3D checkpoint is missing enabled metadata"
    required = ("ddf3d_projection", "ddf3d_router", "adapter")
    missing = [key for key in required if key not in payload]
    if missing:
        return False, f"DDF-3D checkpoint missing components: {missing}"
    adapter = payload["adapter"]
    if not isinstance(adapter, dict) or not any(
        key.startswith("projections.") for key in adapter
    ):
        return False, "DDF-3D adapter state lacks layer-specific projections"
    fusion = metadata.get("fusion_mode")
    if fusion == "patch_softmax" and not any(
        key.startswith("patch_router.") for key in adapter
    ):
        return False, "DDF-3D patch_softmax checkpoint lacks patch router state"
    if fusion == "global_softmax" and "layer_logits" not in adapter:
        return False, "DDF-3D global_softmax checkpoint lacks global logits"
    return True, None


def validate_training_artifacts(checkpoint_path, completion_path):
    checks = (
        validate_checkpoint_file(checkpoint_path),
        validate_yaml_file(completion_path, required_keys=("complete",)),
    )
    errors = [value for valid, value in checks if not valid]
    if not errors and checks[1][1].get("complete") is not True:
        errors.append(f"training completion flag is not true: {completion_path}")
    if (
        not errors
        and checks[1][1].get("use_static_prompt") is True
        and checks[1][1].get("residual_prompt_enabled") is not True
        and "static_prompt" not in checks[0][1]
    ):
        errors.append(f"Prompt checkpoint is missing static_prompt: {checkpoint_path}")
    if (
        not errors
        and checks[1][1].get("residual_prompt_enabled") is True
        and "residual_prompt" not in checks[0][1]
    ):
        errors.append(f"NCRP checkpoint is missing residual_prompt: {checkpoint_path}")
    if (
        not errors
        and checks[1][1].get("ddf3d_enabled") is True
    ):
        ddf_valid, ddf_error = validate_ddf3d_checkpoint_payload(checks[0][1])
        if not ddf_valid:
            errors.append(ddf_error)
    return not errors, errors


def validate_evaluation_artifacts(
    mean_metrics_path,
    per_category_path,
    diagnostics_path=None,
    metrics_path=None,
    sample_scores_path=None,
    completion_path=None,
):
    ncrp = diagnostics_path is not None and "ncrp" in Path(
        diagnostics_path
    ).name.lower()
    checks = [
        validate_json_file(mean_metrics_path, REQUIRED_MEAN_METRICS),
        validate_json_file(per_category_path, required_keys=("metrics",)),
    ]
    if diagnostics_path is not None:
        checks.append(
            validate_json_file(
                diagnostics_path,
                required_keys=(
                    (
                        "num_bases",
                        "basis_usage",
                        "assignment_entropy",
                        "normalized_assignment_entropy",
                        "basis_gram_matrix",
                        "normal_patch_residual_norm",
                        "anomaly_patch_residual_norm",
                        "normal_max_basis_alignment",
                        "anomaly_coverage_cosine",
                        "trainable_parameter_count",
                    )
                    if ncrp
                    else ()
                ),
            )
        )
    if metrics_path is not None:
        checks.append(validate_json_file(metrics_path))
    if sample_scores_path is not None:
        required_sample_arrays = ["patch_logits", "point_scores", "point_labels"]
        if ncrp:
            required_sample_arrays.extend(
                (
                    "labels",
                    "object_labels",
                    "residual_norms",
                    "basis_assignments",
                    "basis_similarities",
                    "combined_residual_direction_norm",
                )
            )
        checks.append(
            validate_npz_file(
                sample_scores_path,
                required_arrays=required_sample_arrays,
            )
        )
    if completion_path is not None:
        checks.append(
            validate_yaml_file(completion_path, required_keys=("complete",))
        )
    errors = [value for valid, value in checks if not valid]
    if completion_path is not None:
        completion_valid, completion = validate_yaml_file(
            completion_path, required_keys=("complete",)
        )
        if completion_valid and completion.get("complete") is not True:
            errors.append(f"completion flag is not true: {completion_path}")
    return not errors, errors


def validate_npz_file(path, required_arrays=()):
    valid, error = _nonempty(path)
    if not valid:
        return False, error
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(set(required_arrays) - set(payload.files))
            if missing:
                return False, f"NPZ missing arrays {missing}: {path}"
    except Exception as exc:
        return False, f"invalid NPZ {path}: {exc}"
    return True, None


def quarantine_corrupt_file(path):
    path = Path(path)
    if not path.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = path.with_name(f"{path.name}.corrupt.{timestamp}")
    path.rename(destination)
    return destination


def _quarantine_invalid(paths, validators):
    quarantined = []
    for path, validator in zip(paths, validators):
        valid, _ = validator(path)
        if not valid and Path(path).exists():
            destination = quarantine_corrupt_file(path)
            quarantined.append((Path(path), destination))
    return quarantined


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("training")
    training.add_argument("--checkpoint", required=True)
    training.add_argument("--completion", required=True)
    evaluation = subparsers.add_parser("evaluation")
    evaluation.add_argument("--mean-metrics", required=True)
    evaluation.add_argument("--per-category", required=True)
    evaluation.add_argument("--diagnostics")
    evaluation.add_argument("--metrics")
    evaluation.add_argument("--sample-scores")
    evaluation.add_argument("--completion")
    for child in (training, evaluation):
        child.add_argument("--quarantine", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "training":
        valid, errors = validate_training_artifacts(
            args.checkpoint, args.completion
        )
        completion_valid, completion_payload = validate_yaml_file(
            args.completion, required_keys=("complete",)
        )
        if completion_valid and completion_payload.get("residual_prompt_enabled") is True:
            checkpoint_keys = ("adapter", "residual_prompt")
        elif completion_valid and completion_payload.get("use_static_prompt") is True:
            checkpoint_keys = ("adapter", "static_prompt")
        else:
            checkpoint_keys = ("adapter",)
        paths = (args.checkpoint, args.completion)
        validators = (
            lambda path: validate_checkpoint_file(path, checkpoint_keys),
            lambda path: validate_yaml_file(path, required_keys=("complete",)),
        )
    elif args.command == "evaluation":
        ncrp = bool(
            args.diagnostics
            and "ncrp" in Path(args.diagnostics).name.lower()
        )
        valid, errors = validate_evaluation_artifacts(
            args.mean_metrics,
            args.per_category,
            args.diagnostics,
            args.metrics,
            args.sample_scores,
            args.completion,
        )
        paths = [args.mean_metrics, args.per_category]
        validators = [
            lambda path: validate_json_file(path, REQUIRED_MEAN_METRICS),
            lambda path: validate_json_file(path, required_keys=("metrics",)),
        ]
        if args.diagnostics:
            paths.append(args.diagnostics)
            validators.append(
                lambda path: validate_json_file(
                    path,
                    required_keys=(
                        (
                            "num_bases",
                            "basis_usage",
                            "assignment_entropy",
                            "normalized_assignment_entropy",
                            "basis_gram_matrix",
                            "normal_patch_residual_norm",
                            "anomaly_patch_residual_norm",
                            "normal_max_basis_alignment",
                            "anomaly_coverage_cosine",
                            "trainable_parameter_count",
                        )
                        if ncrp
                        else ()
                    ),
                )
            )
        if args.metrics:
            paths.append(args.metrics)
            validators.append(validate_json_file)
        if args.sample_scores:
            paths.append(args.sample_scores)
            required_sample_arrays = [
                "patch_logits", "point_scores", "point_labels"
            ]
            if ncrp:
                required_sample_arrays.extend(
                    (
                        "labels",
                        "object_labels",
                        "residual_norms",
                        "basis_assignments",
                        "basis_similarities",
                        "combined_residual_direction_norm",
                    )
                )
            validators.append(
                lambda path: validate_npz_file(
                    path,
                    required_arrays=required_sample_arrays,
                )
            )
        if args.completion:
            paths.append(args.completion)
            validators.append(
                lambda path: validate_yaml_file(path, required_keys=("complete",))
            )
    if valid:
        print(f"Artifact integrity OK: {args.command}")
        return
    for error in errors:
        print(f"WARNING: {error}")
    if args.quarantine:
        for source, destination in _quarantine_invalid(paths, validators):
            print(f"WARNING: quarantined corrupt artifact {source} -> {destination}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
