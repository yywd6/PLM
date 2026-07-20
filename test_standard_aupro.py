"""Evaluate the visual baseline or static Prompts on one-rest targets."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.anomaly_datasets import PointCloudDataset
from evaluation_metrics import finite_mean, point_aupro, safe_binary_metrics
from models.static_prompt import (
    StaticPromptLearner,
    forward_static_prompt_scores,
    point_mask_to_patch_targets as patch_mask_targets,
)
from models.normal_centered_residual_prompt import (
    NormalCenteredResidualPromptLearner,
    forward_ncrp_k1_scores,
)
from models.trainable_baseline import (
    MultiLayerPatchAdapter,
    configurable_object_probability,
    patch_text_logits,
    patch_to_point,
    aggregate_object_probability,
    select_multi_layer_tokens,
)
from models.ddf3d import DDF3DAdapter, forward_ddf3d_fixed_scores
from models.ulip2_encoder import ULIP2Encoder
from one_rest_protocol import (
    assert_no_forbidden_config_keys,
    assert_dataset_categories,
    category_run_name,
    checkpoint_train_categories,
    load_yaml,
    log_line,
    normalize_train_categories,
    resolve_categories,
    validate_one_rest_flags,
    write_json,
    write_yaml,
)
from utils.residual_prompt_config import flatten_residual_prompt_config
from utils.ddf3d_config import (
    add_ddf3d_parser_arguments,
    build_patch_adapter_from_checkpoint,
    flatten_ddf3d_config,
    validate_ddf3d_args,
)
from utils.ddf3d_analysis import (
    DiscrepancyStatistics,
    RoutingStatistics,
    global_routing_weights,
)
from utils.reproducibility import dataloader_seed_kwargs, seed_everything


DEFAULT_CONFIG = "configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml"


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y"}:
        return True
    if value.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the stage-1 visual baseline or stage-2 static Prompts."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", choices=("baseline", "one_rest"), default="baseline")
    parser.add_argument("--train_category")
    parser.add_argument("--train_class", dest="train_category")
    parser.add_argument("--train_categories", nargs="+")
    parser.add_argument("--dataset_name", choices=tuple(PointCloudDataset.PRESETS), default="Real3D")
    parser.add_argument("--data_root")
    parser.add_argument("--test_dataset_name", choices=tuple(PointCloudDataset.PRESETS))
    parser.add_argument("--test_data_root")
    parser.add_argument("--model_path")
    parser.add_argument("--train_split", choices=("train", "test"), default="test")
    parser.add_argument("--test_split", choices=("train", "test"), default="test")
    parser.add_argument("--output_root", default="outputs/trainable_baseline")
    parser.add_argument("--output_dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--return_layers", type=int, nargs="+", default=[2, 5, 8, 11])
    parser.add_argument("--feature_layer", type=int, default=11)
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[2, 5, 8, 11])
    parser.add_argument("--token_dim", type=int, default=384)
    parser.add_argument("--text_dim", type=int, default=1280)
    parser.add_argument("--normal_templates", nargs="+")
    parser.add_argument("--anomaly_templates", nargs="+")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--top_percent", type=float, default=0.2)
    parser.add_argument("--sweep_top_percent", "--sweep_top_p", dest="sweep_top_percent", type=float, nargs="+")
    parser.add_argument("--global_alpha", type=float, default=0.5)
    parser.add_argument(
        "--object_score_mode",
        choices=OBJECT_SCORE_MODES,
        default="aggregate",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_test_samples_per_category", type=int, default=0)
    parser.add_argument("--exclude_train_category_from_test", type=str2bool, default=False)
    parser.add_argument("--zero_shot_target", type=str2bool, default=False)
    parser.add_argument("--use_target_anomaly_for_training", type=str2bool, default=False)
    parser.add_argument("--save_per_category_metrics", type=str2bool, default=True)
    parser.add_argument("--save_mean_metrics", type=str2bool, default=True)
    parser.add_argument("--save_sample_scores", type=str2bool, default=True)
    parser.add_argument("--num_normal_tokens", type=int, default=4)
    parser.add_argument("--num_abnormal_tokens", type=int, default=4)
    parser.add_argument("--use_static_prompt", type=str2bool, default=False)
    parser.add_argument("--use_category_prompt", type=str2bool, default=True)
    parser.add_argument("--prompt_template", default="a point cloud patch of a {category}")
    parser.add_argument("--num_abnormal_prompts", type=int, default=6)
    parser.add_argument("--prompt_score_temperature", type=float, default=0.07)
    parser.add_argument("--static_prompt_version", default="static_six_prompt_v1")
    parser.add_argument("--residual_prompt_enabled", type=str2bool, default=False)
    parser.add_argument("--residual_num_bases", type=int, default=1)
    parser.add_argument("--residual_gamma", type=float, default=1.0)
    parser.add_argument("--residual_eps", type=float, default=1e-6)
    parser.add_argument("--patch_anomaly_threshold", type=float, default=0.05)
    parser.add_argument(
        "--object_pooling_mode",
        choices=("legacy", "top_mean", "top_max", "global_patch_fusion"),
        default="legacy",
    )
    parser.add_argument("--object_top_ratio", type=float, default=0.2)
    parser.add_argument(
        "--source_validation_top_ratios",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.05, 0.2],
    )
    parser.add_argument("--freeze_plm", type=str2bool, default=True)
    parser.add_argument("--freeze_text_encoder", type=str2bool, default=True)
    parser.add_argument("--freeze_visual_adapter", type=str2bool, default=True)
    parser.add_argument(
        "--required_visual_adapter_training_mode",
        choices=("any", "fused_only"),
        default="any",
    )
    parser.add_argument("--baseline_checkpoint")
    parser.add_argument("--prompt_checkpoint")
    add_ddf3d_parser_arguments(parser)
    return parser


def parse_args():
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--config", default=DEFAULT_CONFIG)
    config_args, _ = first.parse_known_args()
    config = flatten_ddf3d_config(
        flatten_residual_prompt_config(load_yaml(config_args.config))
    )
    assert_no_forbidden_config_keys(config, config_args.config)
    parser = build_parser()
    recognized = {action.dest for action in parser._actions}
    parser.set_defaults(
        **{key: value for key, value in config.items() if key in recognized}
    )
    args = parser.parse_args()
    try:
        args.train_categories = normalize_train_categories(
            args.dataset_name, args.train_category, args.train_categories
        )
    except ValueError as error:
        parser.error(str(error))
    args.train_category = category_run_name(args.train_categories)
    if args.num_abnormal_prompts <= 0:
        parser.error("num_abnormal_prompts must be positive")
    if args.prompt_score_temperature <= 0:
        parser.error("prompt_score_temperature must be positive")
    if args.residual_prompt_enabled:
        if not args.use_static_prompt:
            parser.error("NCRP requires frozen text Prompt evaluation")
        if args.residual_num_bases != 1:
            parser.error("NCRP-K1 requires exactly one residual vector")
        if args.residual_gamma < 0 or args.residual_eps <= 0:
            parser.error("NCRP gamma must be non-negative and eps positive")
        if args.object_pooling_mode != "top_mean" or args.object_top_ratio != 0.2:
            parser.error("NCRP v1 fixes top_mean ratio to 0.2")
        if args.global_alpha != 0.0:
            parser.error("NCRP v1 fixes global_alpha=0")
    if not 0 < args.object_top_ratio <= 1:
        parser.error("object_top_ratio must be in (0, 1]")
    validate_one_rest_flags(args)
    validate_ddf3d_args(args, parser)
    return args


def subset_per_category(dataset, limit):
    if limit <= 0:
        return dataset
    selected, counts = [], defaultdict(int)
    for index, category in enumerate(dataset.categories):
        if counts[category] < limit:
            selected.append(index)
            counts[category] += 1
    return Subset(dataset, selected)


LEGACY_STATIC_PROMPT_VERSION = "static_six_mode_prompt_v7_uniform_scoring"
LEGACY_NCRP_K1_VERSION = "ncrp_a1_k1_single_residual"


def checkpoint_uses_static_prompt(checkpoint):
    """Recognize current checkpoints and the completed legacy static run."""
    if "use_static_prompt" in checkpoint:
        return bool(checkpoint["use_static_prompt"])
    return (
        checkpoint.get("geometric_mode_version")
        == LEGACY_STATIC_PROMPT_VERSION
    )


def checkpoint_static_prompt_state(checkpoint):
    if "residual_prompt" in checkpoint:
        return checkpoint["residual_prompt"]
    if "static_prompt" in checkpoint:
        return checkpoint["static_prompt"]
    if "geometric_mode_prompt" in checkpoint:
        return checkpoint["geometric_mode_prompt"]
    raise RuntimeError("Checkpoint does not contain static Prompt weights")


def build_static_prompt_model(args, encoder, checkpoint, device):
    if not checkpoint_uses_static_prompt(checkpoint):
        return None
    checkpoint_version = checkpoint.get(
        "static_prompt_version",
        checkpoint.get("geometric_mode_version"),
    )
    if checkpoint_version not in {
        args.static_prompt_version,
        LEGACY_STATIC_PROMPT_VERSION,
        LEGACY_NCRP_K1_VERSION,
    }:
        raise RuntimeError(
            "Checkpoint Prompt-learning version is incompatible"
        )
    if checkpoint.get("prompt_template") != args.prompt_template:
        raise RuntimeError("Checkpoint prompt_template does not match config")
    checkpoint_residual = bool(checkpoint.get("residual_prompt_enabled", False))
    if checkpoint_residual != bool(args.residual_prompt_enabled):
        raise RuntimeError("Checkpoint residual_prompt_enabled mismatch")
    if checkpoint_residual:
        if int(checkpoint.get("residual_num_bases", 1)) != 1:
            raise RuntimeError("Only NCRP-K1 checkpoints are supported")
        model = NormalCenteredResidualPromptLearner(
            clip_model=encoder.open_clip_model,
            tokenizer=encoder.tokenizer,
            num_bases=int(checkpoint.get("residual_num_bases", args.residual_num_bases)),
            num_normal_tokens=args.num_normal_tokens,
            prompt_template=args.prompt_template,
            use_category_prompt=checkpoint.get(
                "use_category_prompt", args.use_category_prompt
            ),
            gamma=float(checkpoint.get("residual_gamma", args.residual_gamma)),
            eps=float(checkpoint.get("residual_eps", args.residual_eps)),
        ).to(device)
        model.load_state_dict(checkpoint_static_prompt_state(checkpoint), strict=True)
        model.eval()
        return model
    common_kwargs = {
        "clip_model": encoder.open_clip_model,
        "tokenizer": encoder.tokenizer,
        "num_prompts": checkpoint.get(
            "num_abnormal_prompts",
            checkpoint.get("num_geometric_modes", args.num_abnormal_prompts),
        ),
        "num_normal_tokens": args.num_normal_tokens,
        "num_abnormal_tokens": args.num_abnormal_tokens,
        "prompt_template": args.prompt_template,
        "use_category_prompt": checkpoint.get(
            "use_category_prompt", args.use_category_prompt
        ),
    }
    model = StaticPromptLearner(**common_kwargs).to(device)
    model.load_state_dict(checkpoint_static_prompt_state(checkpoint), strict=True)
    model.eval()
    return model


def forward_prompt_scores(
    args,
    adapter,
    layer_tokens,
    global_embeddings,
    prompt_model,
    clip_model,
    object_names,
    patch_centers=None,
):
    if args.residual_prompt_enabled:
        return forward_ncrp_k1_scores(
            adapter,
            layer_tokens,
            global_embeddings,
            prompt_model,
            clip_model,
            object_names,
            temperature=args.temperature,
            patch_centers=patch_centers,
        )
    return forward_static_prompt_scores(
        adapter,
        layer_tokens,
        global_embeddings,
        prompt_model,
        clip_model,
        object_names,
        temperature=args.temperature,
        prompt_score_temperature=args.prompt_score_temperature,
        patch_centers=patch_centers,
    )


def configurable_object_probability_numpy(
    global_logits,
    patch_probabilities,
    mode,
    top_ratio,
    global_alpha,
):
    if mode not in {"top_mean", "top_max", "global_patch_fusion"}:
        raise ValueError(f"Unsupported object pooling mode: {mode}")
    if not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in (0, 1]")
    probabilities = np.asarray(patch_probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("patch_probabilities must have shape [N, G]")
    if mode == "top_max":
        return probabilities.max(axis=1)
    local = _numpy_topk_mean(probabilities, top_ratio, largest=True)
    if mode == "top_mean":
        return local
    global_logits = np.asarray(global_logits, dtype=np.float64).reshape(-1)
    global_probability = 1.0 / (
        1.0 + np.exp(-np.clip(global_logits, -60.0, 60.0))
    )
    return global_alpha * global_probability + (1.0 - global_alpha) * local


def pool_object_scores(args, global_logits, patch_logits, patch_probabilities):
    if args.object_pooling_mode == "legacy":
        return object_scores_from_logits(
            global_logits,
            patch_logits,
            args.global_alpha,
            args.top_percent,
            args.object_score_mode,
        )
    return configurable_object_probability(
        global_logits,
        patch_probabilities,
        args.object_pooling_mode,
        args.object_top_ratio,
        args.global_alpha,
    )


def _torch_topk_mean(values, top_percent, largest=True):
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    count = max(1, int(values.shape[1] * top_percent))
    return values.topk(count, dim=1, largest=largest).values.mean(dim=1)


OBJECT_SCORE_MODES = (
    "aggregate",
    "topk_logit",
    "topk_minus_mean",
    "topk_minus_median",
    "topk_minus_bottom_half",
)

def _numpy_topk_mean(values, top_percent, largest=True):
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must have shape [N, G]")
    count = max(1, int(values.shape[1] * top_percent))
    if largest:
        return np.partition(values, -count, axis=1)[:, -count:].mean(axis=1)
    return np.partition(values, count - 1, axis=1)[:, :count].mean(axis=1)


def aggregate_object_probability_numpy(global_logits, patch_logits, global_alpha=0.5, top_percent=0.2):
    if not 0.0 <= global_alpha <= 1.0:
        raise ValueError("global_alpha must be in [0, 1]")
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    global_logits = np.asarray(global_logits, dtype=np.float64).reshape(-1)
    patch_logits = np.asarray(patch_logits, dtype=np.float64)
    if patch_logits.ndim != 2 or patch_logits.shape[0] != global_logits.shape[0]:
        raise ValueError("patch_logits must have shape [N, G]")
    global_prob = 1.0 / (1.0 + np.exp(-np.clip(global_logits, -60.0, 60.0)))
    patch_prob = 1.0 / (1.0 + np.exp(-np.clip(patch_logits, -60.0, 60.0)))
    local_prob = _numpy_topk_mean(patch_prob, top_percent, largest=True)
    return global_alpha * global_prob + (1.0 - global_alpha) * local_prob


def object_scores_from_logits_numpy(global_logits, patch_logits, global_alpha, top_percent, mode):
    if mode == "aggregate":
        return aggregate_object_probability_numpy(global_logits, patch_logits, global_alpha, top_percent)
    if mode not in OBJECT_SCORE_MODES:
        raise ValueError(f"Unsupported object_score_mode: {mode}")
    patch_logits = np.asarray(patch_logits, dtype=np.float64)
    top_scores = _numpy_topk_mean(patch_logits, top_percent, largest=True)
    if mode == "topk_logit":
        return top_scores
    if mode == "topk_minus_mean":
        return top_scores - patch_logits.mean(axis=1)
    if mode == "topk_minus_median":
        return top_scores - np.median(patch_logits, axis=1)
    if mode == "topk_minus_bottom_half":
        return top_scores - _numpy_topk_mean(patch_logits, 0.5, largest=False)
    raise ValueError(f"Unsupported object_score_mode: {mode}")



def object_scores_from_logits(global_logits, patch_logits, global_alpha, top_percent, mode):
    if mode == "aggregate":
        return aggregate_object_probability(global_logits, patch_logits, global_alpha, top_percent)
    top_scores = _torch_topk_mean(patch_logits, top_percent, largest=True)
    if mode == "topk_logit":
        return top_scores
    if mode == "topk_minus_mean":
        return top_scores - patch_logits.mean(dim=1)
    if mode == "topk_minus_median":
        return top_scores - patch_logits.median(dim=1).values
    if mode == "topk_minus_bottom_half":
        return top_scores - _torch_topk_mean(patch_logits, 0.5, largest=False)
    raise ValueError(f"Unsupported object_score_mode: {mode}")


def v7_fused_point_scores(patch_logits, patch_indices, num_points):
    return patch_to_point(patch_logits, patch_indices, num_points)



def resolve_sweep_top_percent(args):
    if args.sweep_top_percent:
        values = args.sweep_top_percent
    elif args.object_pooling_mode != "legacy":
        values = args.source_validation_top_ratios
    else:
        values = [
            0.000001, 0.00001, 0.0001, 0.0005,
            0.001, 0.002, 0.005, 0.01, 0.02,
            0.05, 0.1, 0.2, 0.3, 0.5, 1.0,
        ]
    valid = sorted({float(value) for value in values if 0 < float(value) <= 1})
    if not valid:
        raise ValueError("sweep_top_percent must contain at least one value in (0, 1]")
    return valid



def _finite_metric(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def normal_false_positive_gap(labels, scores):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    normal = scores[labels <= 0]
    anomaly = scores[labels > 0]
    if normal.size == 0 or anomaly.size == 0:
        return None
    return float(np.quantile(normal, 0.95) - np.quantile(anomaly, 0.50))


def _label_score_mean(labels, scores, positive):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = labels > 0 if positive else labels <= 0
    selected = scores[mask]
    if selected.size == 0:
        return None
    return float(selected.mean())


def build_diagnostic_metrics(args, run_dir, feature_layers, means, best_by_auroc, sample_values):
    object_labels = np.asarray(sample_values["object_label"], dtype=np.int64)
    object_scores = np.asarray(sample_values["object_score"], dtype=np.float64)
    object_auroc = _finite_metric(means.get("object_auroc"))
    point_auroc = _finite_metric(means.get("point_auroc"))
    metrics = {
        "task_name": (
            "ncrp"
            if args.residual_prompt_enabled
            else "static_six_prompt"
        ),
        "layer_set": run_dir.name,
        "layers": list(feature_layers),
        "train_category": args.train_category,
        "dataset_name": args.test_dataset_name or args.dataset_name,
        "source_dataset_name": args.dataset_name,
        "object_auroc": object_auroc,
        "point_auroc": point_auroc,
        "best_top_p": _finite_metric(best_by_auroc.get("top_percent")),
        "best_top_p_object_auroc": _finite_metric(best_by_auroc.get("object_auroc")),
        "normal_false_positive_gap": normal_false_positive_gap(object_labels, object_scores),
        "mean_point_object_gap": (
            point_auroc - object_auroc
            if object_auroc is not None and point_auroc is not None else None
        ),
        "normal_object_score_mean": _label_score_mean(object_labels, object_scores, positive=False),
        "anomaly_object_score_mean": _label_score_mean(object_labels, object_scores, positive=True),
        "object_score_mode": args.object_score_mode,
        "top_percent": args.top_percent,
        "global_alpha": args.global_alpha,
    }
    return metrics


def _fmt_metric(value):
    value = _finite_metric(value)
    return "NA" if value is None else f"{value:.6f}"


def _distribution_stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "std": None, "median": None, "p95": None}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def build_ncrp_diagnostics(
    args,
    checkpoint,
    checkpoint_path,
    means,
    sample_values,
    parameter_counts,
    test_time_seconds,
    inference_time_seconds,
    peak_gpu_memory_bytes,
):
    """Build diagnostics for the maintained single-residual NCRP-K1 method."""
    completion_path = Path(checkpoint_path).parent / "training_complete.yaml"
    completion = load_yaml(completion_path) if completion_path.is_file() else {}
    assignments = np.stack(sample_values["basis_assignments"], axis=0).astype(
        np.float64, copy=False
    )
    residual_norms = np.stack(sample_values["residual_norms"], axis=0).astype(
        np.float64, copy=False
    )
    patch_labels = np.stack(sample_values["patch_labels"], axis=0).astype(bool)
    patch_logits = np.stack(sample_values["patch_logits"], axis=0).astype(
        np.float64, copy=False
    )
    usage = assignments.mean(axis=(0, 1))
    usage = usage / max(float(usage.sum()), args.residual_eps)
    normal_mask = ~patch_labels
    anomaly_mask = patch_labels
    gram_samples = np.stack(sample_values["basis_gram_matrix"], axis=0)
    gram = gram_samples.mean(axis=0)
    offdiag = gram[~np.eye(gram.shape[0], dtype=bool)]
    categories = np.asarray(sample_values["category_name"], dtype=str)
    per_category_usage = {}
    for category in sorted(set(categories.tolist())):
        per_category_usage[category] = assignments[categories == category].mean(
            axis=(0, 1)
        ).tolist()
    training = (checkpoint.get("training_statistics") or {}).get("ncrp") or {}
    diagnostics = {
        "method": "Normal-Centered Residual Prompting (NCRP-K1)",
        "version": checkpoint.get("static_prompt_version"),
        "num_bases": 1,
        "basis_usage": usage.tolist(),
        "assignment_entropy": None,
        "normalized_assignment_entropy": None,
        "assignment_diagnostics_status": "not_applicable_single_basis",
        "basis_usage_status": "structural_single_basis",
        "max_basis_usage": float(usage.max()),
        "max_assignment_weight_mean": float(assignments.max(axis=-1).mean()),
        "basis_gram_matrix": gram.tolist(),
        "basis_gram_off_diagonal_mean": (
            float(np.abs(offdiag).mean()) if offdiag.size else 0.0
        ),
        "basis_normal_anchor_max_abs_inner_product": (
            float(max(sample_values["basis_normal_max_abs_inner_product"]))
            if sample_values["basis_normal_max_abs_inner_product"]
            else None
        ),
        "normal_patch_residual_norm": _distribution_stats(residual_norms[normal_mask]),
        "anomaly_patch_residual_norm": _distribution_stats(residual_norms[anomaly_mask]),
        # Retained as explicit N/A fields for artifact-schema compatibility.
        "normal_max_basis_alignment": None,
        "anomaly_coverage_cosine": None,
        "normal_patch_logit_statistics": _distribution_stats(patch_logits[normal_mask]),
        "anomaly_patch_logit_statistics": _distribution_stats(patch_logits[anomaly_mask]),
        "per_target_category_basis_usage": per_category_usage,
        "trainable_parameter_count": int(parameter_counts),
        "prompt_parameter_counts": checkpoint.get("prompt_parameter_counts"),
        "checkpoint_size_bytes": Path(checkpoint_path).stat().st_size,
        "training_time_seconds": completion.get(
            "training_time_seconds", checkpoint.get("training_time_seconds")
        ),
        "test_time_seconds": float(test_time_seconds),
        "mean_inference_time_per_sample_seconds": float(
            inference_time_seconds / max(1, len(sample_values["path"]))
        ),
        "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
        "gamma": args.residual_gamma,
        "training_curve": training.get("training_curve", []),
        "seed": args.seed,
        "checkpoint_seed": checkpoint.get("seed"),
        "checkpoint_path": str(checkpoint_path),
        "used_target_test_for_training": False,
        "used_target_test_for_hyperparameter_selection": False,
    }
    return diagnostics


@torch.inference_mode()
def main():
    args = parse_args()
    test_started = time.perf_counter()
    seed_report = seed_everything(args.seed, num_workers=args.num_workers)
    train_categories, same_dataset_test_categories = resolve_categories(
        args.dataset_name, args.protocol, train_categories=args.train_categories
    )
    test_dataset_name = args.test_dataset_name or args.dataset_name
    cross_dataset = test_dataset_name != args.dataset_name
    test_categories = (
        list(PointCloudDataset.PRESETS[test_dataset_name])
        if cross_dataset else same_dataset_test_categories
    )
    test_data_root = args.test_data_root or args.data_root
    checkpoint_run_dir = Path(args.output_root) / args.train_category
    run_dir = Path(args.output_dir) if args.output_dir else checkpoint_run_dir
    if (run_dir / "evaluation_complete.yaml").is_file():
        raise FileExistsError(
            f"Refusing to overwrite completed evaluation: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    test_log = run_dir / "test.log"
    test_log.write_text("", encoding="utf-8")
    log_line(test_log, f"Protocol: {args.protocol}")
    log_line(test_log, f"Source dataset: {args.dataset_name}")
    log_line(test_log, f"Test dataset: {test_dataset_name}")
    log_line(test_log, f"Train categories: {train_categories}")
    log_line(test_log, f"Test categories: {test_categories}")
    log_line(test_log, f"Seed report: {seed_report}")
    if args.ddf3d_enabled:
        log_line(
            test_log,
            "DDF-3D enabled"
            f" | fusion={args.ddf3d_fusion_mode}"
            f" | layers={args.ddf3d_layers}"
            f" | top_k={args.ddf3d_router_top_k}",
        )
    sweep_top_percent = resolve_sweep_top_percent(args)
    log_line(test_log, f"Object score mode: {args.object_score_mode}")
    log_line(test_log, f"Object top_p sweep: {sweep_top_percent}")
    log_line(
        test_log,
        f"Object pooling={args.object_pooling_mode}"
        f" | top_ratio={args.object_top_ratio}"
        f" | global_alpha={args.global_alpha}",
    )
    dataset = PointCloudDataset(
        test_data_root, split=args.test_split, classes=test_categories,
        dataset_name=test_dataset_name,
    )
    if len(dataset) == 0:
        raise FileNotFoundError("No target test samples")
    forbidden_categories = (
        train_categories
        if not cross_dataset
        and args.protocol == "one_rest"
        and args.exclude_train_category_from_test
        else ()
    )
    observed = assert_dataset_categories(dataset, test_categories, forbidden_categories)
    log_line(test_log, f"Observed test path categories: {observed}")
    limit = args.max_test_samples_per_category
    if args.debug:
        limit = limit or 4
        log_line(test_log, f"Debug mode: max_test_samples_per_category={limit}")
    loader = DataLoader(
        subset_per_category(dataset, limit), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        **dataloader_seed_kwargs(args.seed),
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_run_dir / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint_train_categories(checkpoint) != args.train_categories:
        raise RuntimeError("Checkpoint train_categories mismatch")
    if "feature_layers" not in checkpoint:
        raise RuntimeError(
            "Checkpoint uses the old single-layer adapter; retrain with feature_layers=[2,5,8,11]"
        )
    visual_source_training_mode = (
        checkpoint.get("visual_source_training_mode")
        or checkpoint.get("visual_adapter_training_mode")
        or "fused_only"
    )
    if (
        args.required_visual_adapter_training_mode != "any"
        and visual_source_training_mode
        != args.required_visual_adapter_training_mode
    ):
        raise RuntimeError(
            "Visual adapter training mode mismatch: "
            f"required={args.required_visual_adapter_training_mode}, "
            f"checkpoint={visual_source_training_mode}"
        )
    log_line(
        test_log,
        f"Visual adapter training mode: {visual_source_training_mode}",
    )
    checkpoint_feature_layers = list(checkpoint.get("feature_layers", args.feature_layers))
    feature_layers = checkpoint_feature_layers
    args.return_layers = list(feature_layers)
    args.feature_layers = list(feature_layers)
    log_line(test_log, f"Checkpoint PointBERT feature_layers: {checkpoint_feature_layers}")
    log_line(test_log, f"Evaluation PointBERT feature_layers: {feature_layers}")
    use_static_prompt = checkpoint_uses_static_prompt(checkpoint)
    if use_static_prompt:
        if args.residual_prompt_enabled:
            log_line(test_log, "Prompt learner=NCRP")
            log_line(test_log, f"Learnable residual bases={args.residual_num_bases}")
        else:
            prompt_count = checkpoint.get(
                "num_abnormal_prompts",
                checkpoint.get("num_geometric_modes", args.num_abnormal_prompts),
            )
            log_line(test_log, "StaticPromptLearner=True")
            log_line(test_log, f"Learnable abnormal Prompts={prompt_count}")
        log_line(test_log, f"Prompt template: {args.prompt_template}")
    else:
        log_line(test_log, "Stage 1: trainable multi-layer visual baseline")
    encoder = ULIP2Encoder(
        args.model_path, device=device, num_points=args.num_points,
        return_layers=tuple(args.return_layers), return_clip=use_static_prompt,
    )
    log_line(
        test_log,
        "PointBERT blocks="
        f"{encoder.pointbert_block_count} | configured layer -> Python index="
        f"{encoder.layer_block_indices}",
    )
    adapter = build_patch_adapter_from_checkpoint(args, checkpoint).to(device)
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.eval()
    ddf3d_state_before = (
        {
            key: value.detach().cpu().clone()
            for key, value in adapter.state_dict().items()
        }
        if isinstance(adapter, DDF3DAdapter)
        else None
    )
    prompt_model = build_static_prompt_model(args, encoder, checkpoint, device)
    trainable_parameter_count = (
        sum(parameter.numel() for parameter in prompt_model.parameters())
        if prompt_model is not None
        else sum(parameter.numel() for parameter in adapter.parameters())
    )
    if prompt_model is None:
        fixed_normal = encoder.encode_text_templates(args.normal_templates)
        fixed_anomaly = encoder.encode_text_templates(args.anomaly_templates)
    else:
        fixed_normal = fixed_anomaly = None

    values = defaultdict(lambda: defaultdict(list))
    sample_values = defaultdict(list)
    routing_statistics = {
        name: RoutingStatistics(adapter.layers)
        for name in test_categories
    } if isinstance(adapter, DDF3DAdapter) else {}
    discrepancy_statistics = {
        name: DiscrepancyStatistics() for name in test_categories
    } if isinstance(adapter, DDF3DAdapter) else {}
    inference_time_seconds = 0.0
    category_names = PointCloudDataset.PRESETS[test_dataset_name]
    for batch in tqdm(loader, desc=f"Test source={args.train_category}", dynamic_ncols=True):
        if device == "cuda":
            torch.cuda.synchronize()
        inference_started = time.perf_counter()
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"]
        object_names = [category_names[int(category)] for category in batch["category"]]
        features = encoder.encode_pointcloud(points, return_intermediate=True)
        tokens = select_multi_layer_tokens(
            features.get("patch_tokens", features["layer_feats"]),
            features["patch_idx"],
            feature_layers,
        )
        if prompt_model is not None:
            score_output = forward_prompt_scores(
                args,
                adapter,
                tokens,
                features["concat"],
                prompt_model,
                encoder.open_clip_model,
                object_names,
                patch_centers=features["patch_centers"],
            )
            patch_embeddings = score_output["patch_embeddings"]
            patch_logits = score_output["patch_logits"]
            global_logits = score_output["global_logits"]
        else:
            if isinstance(adapter, DDF3DAdapter):
                score_output = forward_ddf3d_fixed_scores(
                    adapter,
                    tokens,
                    features["patch_centers"],
                    features["concat"],
                    fixed_normal,
                    fixed_anomaly,
                    args.temperature,
                )
                patch_embeddings = score_output["patch_embeddings"]
                patch_logits = score_output["patch_logits"]
                global_logits = score_output["global_logits"]
            else:
                patch_embeddings = adapter(tokens)
                patch_logits = patch_text_logits(
                    patch_embeddings, fixed_normal, fixed_anomaly, args.temperature
                )
                global_logits = patch_text_logits(
                    features["concat"].unsqueeze(1),
                    fixed_normal,
                    fixed_anomaly,
                    args.temperature,
                ).squeeze(1)
        patch_probabilities = torch.sigmoid(patch_logits.double())
        point_scores_tensor = patch_to_point(
            patch_logits, features["patch_idx"], labels.shape[1]
        )
        point_scores = point_scores_tensor.cpu().numpy()
        object_scores_for_metrics_tensor = pool_object_scores(
            args,
            global_logits,
            patch_logits,
            patch_probabilities,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        inference_time_seconds += time.perf_counter() - inference_started
        object_scores = object_scores_for_metrics_tensor.cpu().numpy()
        global_logit_values = global_logits.detach().cpu().numpy()
        patch_logit_values = patch_logits.detach().cpu().numpy()
        patch_probability_values = patch_probabilities.cpu().numpy()
        points_for_save = points.detach().cpu().numpy()
        if args.residual_prompt_enabled or isinstance(adapter, DDF3DAdapter):
            patch_anomaly_mask, _, _, _ = patch_mask_targets(
                labels.to(device),
                features["patch_idx"],
                args.patch_anomaly_threshold,
            )
            patch_label_values = patch_anomaly_mask.detach().cpu().numpy()
        if args.residual_prompt_enabled:
            basis_assignment_values = score_output["basis_assignments"].detach().cpu().numpy()
            basis_similarity_values = score_output["basis_similarities"].detach().cpu().numpy()
            residual_norm_values = score_output["residual_norms"].detach().cpu().numpy()
            combined_norm_values = score_output[
                "combined_residual_direction_norm"
            ].detach().cpu().numpy()
            directions = score_output["projected_directions"]
            basis_gram_values = (
                directions @ directions.transpose(-1, -2)
            ).detach().cpu().numpy()
            basis_normal_inner_values = score_output.get(
                "basis_normal_max_abs_inner_product"
            )
            basis_normal_inner_values = (
                basis_normal_inner_values.detach().cpu().numpy()
                if basis_normal_inner_values is not None
                else None
            )
        paths = batch.get("path", [""] * len(batch["category"]))
        for index, category in enumerate(batch["category"]):
            name = category_names[int(category)]
            if isinstance(adapter, DDF3DAdapter):
                ddf3d_output = score_output.get("ddf3d", score_output)
                routing_statistics[name].update(
                    ddf3d_output["routing_weights"][index : index + 1]
                )
                discrepancy_statistics[name].update(
                    {
                        key: value[index : index + 1]
                        for key, value in ddf3d_output.items()
                        if torch.is_tensor(value)
                    },
                    patch_anomaly_mask[index : index + 1],
                )
            point_labels = labels[index].numpy().reshape(-1)
            object_label = int((point_labels > 0).any())
            values[name]["object_labels"].append(object_label)
            values[name]["object_scores"].append(float(object_scores[index]))
            values[name]["global_logits"].append(float(global_logit_values[index]))
            values[name]["patch_logits"].append(patch_logit_values[index].copy())
            values[name]["point_labels"].append(point_labels)
            values[name]["point_scores"].append(point_scores[index].reshape(-1))
            values[name]["patch_probabilities"].append(
                patch_probability_values[index].copy()
            )
            sample_values["path"].append(str(paths[index]))
            sample_values["category_name"].append(name)
            sample_values["category_index"].append(int(category))
            sample_values["object_label"].append(object_label)
            sample_values["object_score"].append(float(object_scores[index]))
            sample_values["global_logit"].append(float(global_logit_values[index]))
            sample_values["patch_logits"].append(patch_logit_values[index].copy())
            if args.residual_prompt_enabled:
                sample_values["patch_labels"].append(patch_label_values[index].copy())
                sample_values["basis_assignments"].append(
                    basis_assignment_values[index].copy()
                )
                sample_values["basis_similarities"].append(
                    basis_similarity_values[index].copy()
                )
                sample_values["residual_norms"].append(
                    residual_norm_values[index].copy()
                )
                sample_values["combined_residual_direction_norm"].append(
                    combined_norm_values[index].copy()
                )
                sample_values["basis_gram_matrix"].append(
                    basis_gram_values[index].copy()
                )
                if basis_normal_inner_values is not None:
                    sample_values["basis_normal_max_abs_inner_product"].append(
                        float(basis_normal_inner_values[index])
                    )
            sample_values["point_scores"].append(point_scores[index].reshape(-1).copy())
            sample_values["point_labels"].append(point_labels.copy())
            sample_values["points"].append(points_for_save[index].copy())
            sample_values["patch_probabilities"].append(
                patch_probability_values[index].copy()
            )

    if isinstance(adapter, DDF3DAdapter):
        for target_name in test_categories:
            if adapter.settings.fusion_mode == "global_softmax":
                routing_payload = global_routing_weights(
                    adapter.layers, adapter.layer_weights()
                )
            else:
                routing_payload = routing_statistics[target_name].result()
            routing_path = (
                run_dir
                / "routing_stats"
                / args.train_category
                / f"{target_name}.json"
            )
            discrepancy_path = (
                run_dir
                / "discrepancy_stats"
                / args.train_category
                / f"{target_name}.json"
            )
            write_json(routing_path, routing_payload)
            write_json(
                discrepancy_path,
                discrepancy_statistics[target_name].result(),
            )
        log_line(
            test_log,
            "Saved compact DDF-3D routing/discrepancy statistics for "
            f"{len(test_categories)} target categories",
        )

    if args.save_sample_scores:
        score_payload = {
            "path": np.asarray(sample_values["path"], dtype=str),
            "category_name": np.asarray(sample_values["category_name"], dtype=str),
            "category_index": np.asarray(sample_values["category_index"], dtype=np.int64),
            "object_label": np.asarray(sample_values["object_label"], dtype=np.int64),
            "object_labels": np.asarray(sample_values["object_label"], dtype=np.int64),
            "object_score": np.asarray(sample_values["object_score"], dtype=np.float64),
            "global_logit": np.asarray(sample_values["global_logit"], dtype=np.float64),
            "patch_logits": np.stack(sample_values["patch_logits"], axis=0),
            "patch_probabilities": np.stack(
                sample_values["patch_probabilities"], axis=0
            ),
            "point_scores": np.stack(sample_values["point_scores"], axis=0),
            "point_labels": np.stack(sample_values["point_labels"], axis=0),
            "labels": np.stack(sample_values["point_labels"], axis=0),
            "points": np.stack(sample_values["points"], axis=0),
            "object_score_mode": np.asarray(args.object_score_mode),
            "top_percent": np.asarray(args.top_percent, dtype=np.float64),
            "object_top_ratio": np.asarray(args.object_top_ratio, dtype=np.float64),
            "object_pooling_mode": np.asarray(args.object_pooling_mode),
            "global_alpha": np.asarray(args.global_alpha, dtype=np.float64),
            "pointbert_layers": np.asarray(feature_layers, dtype=np.int64),
        }
        if args.residual_prompt_enabled:
            score_payload.update(
                {
                    "residual_norms": np.stack(
                        sample_values["residual_norms"], axis=0
                    ),
                    "basis_assignments": np.stack(
                        sample_values["basis_assignments"], axis=0
                    ),
                    "basis_similarities": np.stack(
                        sample_values["basis_similarities"], axis=0
                    ),
                    "combined_residual_direction_norm": np.stack(
                        sample_values["combined_residual_direction_norm"], axis=0
                    ),
                    "patch_labels": np.stack(
                        sample_values["patch_labels"], axis=0
                    ),
                }
            )
            if sample_values["basis_normal_max_abs_inner_product"]:
                score_payload["basis_normal_max_abs_inner_product"] = np.asarray(
                    sample_values["basis_normal_max_abs_inner_product"],
                    dtype=np.float64,
                )
        np.savez_compressed(run_dir / "sample_scores.npz", **score_payload)
        log_line(test_log, f"Saved sample scores: {run_dir / 'sample_scores.npz'}")

    per_category = {}
    for name in test_categories:
        category = values[name]
        object_metrics = safe_binary_metrics(category["object_labels"], category["object_scores"])
        point_labels = np.concatenate(category["point_labels"]) if category["point_labels"] else np.array([])
        point_scores = np.concatenate(category["point_scores"]) if category["point_scores"] else np.array([])
        point_metrics = safe_binary_metrics(point_labels, point_scores)
        per_category[name] = {
            "samples": len(category["object_labels"]),
            "object_auroc": object_metrics["auroc"], "object_ap": object_metrics["ap"],
            "point_auroc": point_metrics["auroc"], "point_ap": point_metrics["ap"],
            "point_pro": point_aupro(category["point_scores"], category["point_labels"]),
        }
        log_line(test_log, f"{name}: {per_category[name]}")

    metric_names = (
        "object_auroc", "object_ap",
        "point_auroc", "point_ap", "point_pro",
    )
    means = {key: finite_mean([item.get(key) for item in per_category.values()]) for key in metric_names}
    sweep_results = []
    for top_percent in sweep_top_percent:
        sweep_per_category = {}
        for name in test_categories:
            category = values[name]
            object_labels = category["object_labels"]
            global_logits_np = np.asarray(category["global_logits"], dtype=np.float64)
            patch_logits_np = np.stack(category["patch_logits"], axis=0)
            if args.object_pooling_mode == "legacy":
                object_scores_np = object_scores_from_logits_numpy(
                    global_logits_np, patch_logits_np, args.global_alpha,
                    top_percent, args.object_score_mode,
                )
            else:
                patch_probabilities_np = np.stack(
                    category["patch_probabilities"], axis=0
                )
                object_scores_np = configurable_object_probability_numpy(
                    global_logits_np,
                    patch_probabilities_np,
                    (
                        "global_patch_fusion"
                        if args.object_pooling_mode == "legacy"
                        else args.object_pooling_mode
                    ),
                    top_percent,
                    args.global_alpha,
                )
            object_metrics = safe_binary_metrics(object_labels, object_scores_np.tolist())
            sweep_per_category[name] = {
                "samples": len(object_labels),
                "object_auroc": object_metrics["auroc"],
                "object_ap": object_metrics["ap"],
            }
        sweep_mean = {
            "object_auroc": finite_mean(
                [item["object_auroc"] for item in sweep_per_category.values()]
            ),
            "object_ap": finite_mean(
                [item["object_ap"] for item in sweep_per_category.values()]
            ),
        }
        sweep_results.append({
            "top_percent": top_percent,
            "global_alpha": args.global_alpha,
            "object_score_mode": args.object_score_mode,
            "object_pooling_mode": args.object_pooling_mode,
            **sweep_mean,
            "per_category": sweep_per_category,
        })
        log_line(
            test_log,
            f"Top-p sweep top_percent={top_percent:g}: "
            f"object_auroc={_fmt_metric(sweep_mean['object_auroc'])}, "
            f"object_ap={_fmt_metric(sweep_mean['object_ap'])}",
        )
    if args.object_pooling_mode == "legacy":
        best_by_auroc = max(
            sweep_results,
            key=lambda item: item["object_auroc"] if item["object_auroc"] is not None else float("-inf"),
        )
        best_by_ap = max(
            sweep_results,
            key=lambda item: item["object_ap"] if item["object_ap"] is not None else float("-inf"),
        )
    else:
        configured = min(
            sweep_results,
            key=lambda item: abs(item["top_percent"] - args.object_top_ratio),
        )
        # Never choose a new-method ratio using target-test labels. Keep this
        # alias only for the existing diagnostic report interface.
        best_by_auroc = configured
        best_by_ap = configured
    sweep_summary = {
        "selection_policy": (
            "diagnostic_only; final top ratio must be selected on source-only "
            "validation, never on target-test metrics"
        ),
        "train_category": args.train_category,
        "source_dataset_name": args.dataset_name,
        "test_dataset_name": test_dataset_name,
        "test_categories": test_categories,
        "global_alpha": args.global_alpha,
        "object_score_mode": args.object_score_mode,
        "object_pooling_mode": args.object_pooling_mode,
        "results": sweep_results,
    }
    if args.object_pooling_mode == "legacy":
        sweep_summary["best_by_object_auroc"] = {
            key: best_by_auroc[key]
            for key in ("top_percent", "object_auroc", "object_ap")
        }
        sweep_summary["best_by_object_ap"] = {
            key: best_by_ap[key]
            for key in ("top_percent", "object_auroc", "object_ap")
        }
    else:
        sweep_summary["configured_ratio_result"] = {
            key: best_by_auroc[key]
            for key in ("top_percent", "object_auroc", "object_ap")
        }
    write_json(run_dir / "object_top_percent_sweep.json", sweep_summary)
    if args.object_pooling_mode != "legacy":
        write_json(run_dir / "object_top_ratio_sweep.json", sweep_summary)
    if args.object_pooling_mode == "legacy":
        log_line(
            test_log,
            "Best top-p by O-AUROC: "
            f"top_percent={best_by_auroc['top_percent']:g}, "
            f"object_auroc={_fmt_metric(best_by_auroc['object_auroc'])}, "
            f"object_ap={_fmt_metric(best_by_auroc['object_ap'])}",
        )
        log_line(
            test_log,
            "Best top-p by O-AP: "
            f"top_percent={best_by_ap['top_percent']:g}, "
            f"object_auroc={_fmt_metric(best_by_ap['object_auroc'])}, "
            f"object_ap={_fmt_metric(best_by_ap['object_ap'])}",
        )
    else:
        log_line(
            test_log,
            "Configured source-selected top ratio: "
            f"top_ratio={best_by_auroc['top_percent']:g}, "
            f"object_auroc={_fmt_metric(best_by_auroc['object_auroc'])}, "
            f"object_ap={_fmt_metric(best_by_auroc['object_ap'])}",
        )
    if args.residual_prompt_enabled:
        test_time_seconds = time.perf_counter() - test_started
        peak_memory = (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
        )
        ncrp_diagnostics = build_ncrp_diagnostics(
            args,
            checkpoint,
            checkpoint_path,
            means,
            sample_values,
            trainable_parameter_count,
            test_time_seconds,
            inference_time_seconds,
            peak_memory,
        )
        sample_path = run_dir / "sample_scores.npz"
        ncrp_diagnostics["sample_scores_size_bytes"] = (
            sample_path.stat().st_size if sample_path.is_file() else None
        )
        diagnostics_path = run_dir / "ncrp_diagnostics.json"
        write_json(diagnostics_path, ncrp_diagnostics)
        ncrp_diagnostics["diagnostics_size_bytes"] = diagnostics_path.stat().st_size
        write_json(diagnostics_path, ncrp_diagnostics)
        log_line(test_log, f"NCRP diagnostics: {diagnostics_path}")
    diagnostic_metrics = build_diagnostic_metrics(
        args, run_dir, feature_layers, means, best_by_auroc, sample_values,
    )
    write_json(run_dir / "metrics.json", diagnostic_metrics)
    log_line(test_log, f"Diagnostic metrics: {run_dir / 'metrics.json'}")
    if args.save_per_category_metrics:
        write_json(run_dir / "per_category_metrics.json", {
            "train_category": args.train_category, "test_categories": test_categories,
            "source_dataset_name": args.dataset_name,
            "test_dataset_name": test_dataset_name,
            "use_static_prompt": use_static_prompt,
            "object_pooling_mode": args.object_pooling_mode,
            "object_top_ratio": args.object_top_ratio,
            "residual_prompt_enabled": args.residual_prompt_enabled,
            "seed": args.seed,
            "seed_report": seed_report,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_seed": checkpoint.get("seed"),
            "metrics": per_category,
        })
    if args.save_mean_metrics:
        write_json(run_dir / "mean_metrics.json", {
            "train_category": args.train_category, "test_categories": test_categories, **means,
            "source_dataset_name": args.dataset_name,
            "test_dataset_name": test_dataset_name,
            "object_pooling_mode": args.object_pooling_mode,
            "object_top_ratio": args.object_top_ratio,
            "residual_prompt_enabled": args.residual_prompt_enabled,
            "seed": args.seed,
            "seed_report": seed_report,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_seed": checkpoint.get("seed"),
        })
    log_line(
        test_log,
        "Mean metrics: "
        f"object_auroc={means.get('object_auroc')} "
        f"point_auroc={means.get('point_auroc')} "
        f"all={means}",
    )
    if args.residual_prompt_enabled:
        log_line(
            test_log,
            "NCRP test resources"
            f" | test_time_seconds={time.perf_counter() - test_started:.3f}"
            " | mean_inference_time_per_sample_seconds="
            f"{inference_time_seconds / max(1, len(sample_values['path'])):.6f}"
            " | peak_gpu_memory_bytes="
            f"{int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0}",
        )
    if isinstance(adapter, DDF3DAdapter):
        state_after = adapter.state_dict()
        changed = [
            key
            for key, before in ddf3d_state_before.items()
            if not torch.equal(before, state_after[key].detach().cpu())
        ]
        if changed:
            raise RuntimeError(
                f"DDF-3D parameters changed during target evaluation: {changed}"
            )
    write_yaml(
        run_dir / "evaluation_complete.yaml",
        {
            "complete": True,
            "ddf3d_enabled": isinstance(adapter, DDF3DAdapter),
            "fusion_mode": (
                adapter.settings.fusion_mode
                if isinstance(adapter, DDF3DAdapter)
                else None
            ),
            "train_category": args.train_category,
            "test_categories": list(test_categories),
            "checkpoint_path": str(checkpoint_path),
            "router_updated_during_test": (
                False if isinstance(adapter, DDF3DAdapter) else None
            ),
            "test_time_seconds": time.perf_counter() - test_started,
            "inference_time_seconds": inference_time_seconds,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if device == "cuda"
                else 0
            ),
        },
    )


if __name__ == "__main__":
    main()
