"""Two-stage training for the visual baseline and static Prompt learning."""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from data.anomaly_datasets import PointCloudDataset
from evaluation_metrics import safe_binary_metrics
from loss import BinaryDiceLoss, BinaryFocalLoss
from models.static_prompt import (
    StaticPromptLearner,
    forward_static_prompt_scores,
    format_category_prompt,
    point_mask_to_patch_targets,
    prompt_diversity_loss,
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
    build_patch_adapter,
    flatten_ddf3d_config,
    validate_checkpoint_ddf3d_config,
    validate_ddf3d_args,
)
from utils.reproducibility import dataloader_seed_kwargs, seed_everything
from utils.ddf3d_analysis import (
    RoutingStatistics,
    global_routing_weights,
)


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
        description="Train the stage-1 visual baseline or stage-2 static Prompts."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", choices=("baseline", "one_rest"), default="baseline")
    parser.add_argument("--train_category")
    parser.add_argument("--train_class", dest="train_category")
    parser.add_argument("--train_categories", nargs="+")
    parser.add_argument("--dataset_name", choices=tuple(PointCloudDataset.PRESETS), default="Real3D")
    parser.add_argument("--data_root")
    parser.add_argument("--model_path")
    parser.add_argument("--train_split", choices=("train", "test"), default="test")
    parser.add_argument("--test_split", choices=("train", "test"), default="test")
    parser.add_argument("--output_root", default="outputs/trainable_baseline")
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
    parser.add_argument("--global_alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--prompt_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--minimum_learning_rate", type=float, default=1e-6)
    parser.add_argument("--gradient_clip_norm", type=float, default=0.0)
    parser.add_argument("--validation_fraction", type=float, default=0.0)
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--checkpoint_metric", choices=("loss", "val_object_auroc"), default="loss")
    parser.add_argument("--focal_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--object_weight", type=float, default=0.5)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--exclude_train_category_from_test", type=str2bool, default=False)
    parser.add_argument("--zero_shot_target", type=str2bool, default=False)
    parser.add_argument("--use_target_anomaly_for_training", type=str2bool, default=False)
    parser.add_argument("--save_per_category_metrics", type=str2bool, default=True)
    parser.add_argument("--save_mean_metrics", type=str2bool, default=True)
    parser.add_argument("--num_normal_tokens", type=int, default=4)
    parser.add_argument("--num_abnormal_tokens", type=int, default=4)
    parser.add_argument("--use_static_prompt", type=str2bool, default=False)
    parser.add_argument("--use_category_prompt", type=str2bool, default=True)
    parser.add_argument("--prompt_template", default="a point cloud patch of a {category}")
    parser.add_argument("--num_abnormal_prompts", type=int, default=6)
    parser.add_argument("--prompt_score_temperature", type=float, default=0.07)
    parser.add_argument("--static_prompt_version", default="static_six_prompt_v1")
    parser.add_argument("--use_prompt_diversity_loss", type=str2bool, default=True)
    parser.add_argument("--lambda_prompt_diversity", type=float, default=0.01)
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
        "--visual_adapter_training_mode",
        choices=("fused_only",),
        default="fused_only",
    )
    parser.add_argument(
        "--required_visual_adapter_training_mode",
        choices=("any", "fused_only"),
        default="any",
    )
    parser.add_argument("--baseline_checkpoint")
    parser.add_argument("--prompt_checkpoint")
    parser.add_argument("--resume_checkpoint")
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
    valid = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    parser.set_defaults(**config)
    args = parser.parse_args()
    try:
        args.train_categories = normalize_train_categories(
            args.dataset_name, args.train_category, args.train_categories
        )
    except ValueError as error:
        parser.error(str(error))
    args.train_category = category_run_name(args.train_categories)
    missing_layers = [
        layer for layer in args.feature_layers if layer not in args.return_layers
    ]
    if missing_layers:
        parser.error(
            f"feature_layers must be included in return_layers: {missing_layers}"
        )
    if args.num_abnormal_prompts <= 0:
        parser.error("num_abnormal_prompts must be positive")
    if args.prompt_score_temperature <= 0:
        parser.error("prompt_score_temperature must be positive")
    if args.residual_prompt_enabled:
        if not args.use_static_prompt:
            parser.error("NCRP requires the frozen text Prompt training path")
        if args.residual_num_bases != 1:
            parser.error("NCRP-K1 requires exactly one residual vector")
        if args.use_prompt_diversity_loss:
            parser.error("NCRP-K1 does not use Prompt diversity loss")
        if args.object_pooling_mode != "top_mean" or args.object_top_ratio != 0.2:
            parser.error("NCRP v1 fixes object pooling to top_mean with ratio 0.2")
        if args.global_alpha != 0.0:
            parser.error("NCRP v1 fixes global_alpha=0")
    if args.residual_eps <= 0 or args.residual_gamma < 0:
        parser.error("NCRP gamma must be non-negative and eps positive")
    if not 0 <= args.patch_anomaly_threshold <= 1:
        parser.error("patch_anomaly_threshold must be in [0, 1]")
    if not 0 < args.object_top_ratio <= 1:
        parser.error("object_top_ratio must be in (0, 1]")
    if not args.source_validation_top_ratios or any(
        not 0 < value <= 1 for value in args.source_validation_top_ratios
    ):
        parser.error("source_validation_top_ratios must be in (0, 1]")
    if args.use_static_prompt and not (
        args.freeze_plm and args.freeze_text_encoder
    ):
        parser.error("Prompt learning requires frozen PLM and text encoder")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("warmup_ratio must be in [0, 1)")
    if args.use_static_prompt and (
        args.visual_adapter_training_mode != "fused_only"
    ):
        parser.error(
            "visual_adapter_training_mode applies only to stage-1 visual training"
        )
    if args.minimum_learning_rate < 0 or args.gradient_clip_norm < 0:
        parser.error(
            "minimum_learning_rate and gradient_clip_norm must be non-negative"
        )
    if not 0 <= args.validation_fraction < 1:
        parser.error("validation_fraction must be in [0, 1)")
    if args.validation_interval <= 0:
        parser.error("validation_interval must be positive")
    if (
        args.checkpoint_metric == "val_object_auroc"
        and args.validation_fraction <= 0
    ):
        parser.error(
            "checkpoint_metric=val_object_auroc requires validation_fraction > 0"
        )
    validate_one_rest_flags(args)
    validate_ddf3d_args(args, parser)
    return args


def set_seed(seed, num_workers=0):
    return seed_everything(seed, num_workers=num_workers)



def stratified_holdout_indices(dataset, candidate_indices, fraction, seed):
    if fraction <= 0 or len(candidate_indices) < 2:
        return list(candidate_indices), []
    rng = random.Random(seed)
    grouped = {0: [], 1: []}
    for idx in candidate_indices:
        sample = dataset[idx]
        label = int(sample["labels"].max().item() > 0)
        grouped[label].append(idx)
    train_indices, val_indices = [], []
    for group in grouped.values():
        if not group:
            continue
        rng.shuffle(group)
        if len(group) == 1:
            train_indices.extend(group)
            continue
        val_count = max(1, int(round(len(group) * fraction)))
        val_count = min(val_count, len(group) - 1)
        val_indices.extend(group[:val_count])
        train_indices.extend(group[val_count:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    if not train_indices or not val_indices:
        return list(candidate_indices), []
    return train_indices, val_indices


def build_checkpoint_state(
    args, adapter, prompt_model, epoch, loss, selection_metric, selection_value
):
    state = {
        "adapter": adapter.state_dict(),
        "train_category": args.train_category,
        "train_categories": list(args.train_categories),
        "feature_layers": list(args.feature_layers),
        "token_dim": args.token_dim,
        "text_dim": args.text_dim,
        "epoch": epoch,
        "loss": float(loss),
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
        "checkpoint_metric": args.checkpoint_metric,
        "seed": args.seed,
        "seed_report": getattr(args, "seed_report", None),
        "training_time_seconds": (
            getattr(args, "training_time_offset", 0.0)
            + time.perf_counter() - args.training_started
            if hasattr(args, "training_started")
            else None
        ),
        "validation_fraction": args.validation_fraction,
        "validation_interval": args.validation_interval,
        "prompt_checkpoint": args.prompt_checkpoint,
        "use_static_prompt": args.use_static_prompt,
        "static_prompt_version": (
            args.static_prompt_version if args.use_static_prompt else None
        ),
        "num_abnormal_prompts": (
            args.num_abnormal_prompts if args.use_static_prompt else None
        ),
        "prompt_template": (
            args.prompt_template if args.use_static_prompt else None
        ),
        "use_category_prompt": (
            args.use_category_prompt if args.use_static_prompt else None
        ),
        "prompt_score_temperature": (
            args.prompt_score_temperature if args.use_static_prompt else None
        ),
        "patch_anomaly_threshold": args.patch_anomaly_threshold,
        "object_pooling_mode": args.object_pooling_mode,
        "object_top_ratio": args.object_top_ratio,
        "source_validation_top_ratios": list(args.source_validation_top_ratios),
        "visual_adapter_training_mode": args.visual_adapter_training_mode,
        "required_visual_adapter_training_mode": (
            args.required_visual_adapter_training_mode
        ),
        "visual_source_training_mode": getattr(
            args, "visual_source_training_mode", None
        ),
        "residual_prompt_enabled": args.residual_prompt_enabled,
        "residual_num_bases": args.residual_num_bases,
        "residual_gamma": args.residual_gamma,
        "residual_eps": args.residual_eps,
    }
    if prompt_model is not None:
        state[
            "residual_prompt" if args.residual_prompt_enabled else "static_prompt"
        ] = prompt_model.state_dict()
    if isinstance(adapter, DDF3DAdapter):
        state["ddf3d"] = adapter.checkpoint_metadata()
        state["ddf3d_projection"] = adapter.projection_state_dict()
        state["ddf3d_router"] = adapter.router_state_dict()
    return state


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
    """Route the two maintained Prompt methods: NCRP-K1 and Static."""
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


def pool_training_object_probability(args, global_logits, patch_logits):
    if args.object_pooling_mode == "legacy":
        return aggregate_object_probability(
            global_logits, patch_logits, args.global_alpha, args.top_percent
        )
    return configurable_object_probability(
        global_logits,
        torch.sigmoid(patch_logits.double()),
        args.object_pooling_mode,
        args.object_top_ratio,
        args.global_alpha,
    )


def prompt_parameter_counts(prompt_model):
    if isinstance(prompt_model, NormalCenteredResidualPromptLearner):
        bank = prompt_model.prompt_bank
        counts = {
            "normal_prompt_tokens": bank.normal_tokens.numel(),
            "local_residual_basis": bank.local_residual_basis.numel(),
            "abnormal_prompt_tokens": 0,
        }
        counts["total"] = sum(counts.values())
        return counts
    counts = {
        "prompt_tokens": sum(
            parameter.numel()
            for parameter in prompt_model.prompt_bank.parameters()
        ),
    }
    counts["total"] = sum(counts.values())
    return counts


@torch.no_grad()
def evaluate_holdout_object_metrics(
    args, loader, encoder, adapter, prompt_model, fixed_normal, fixed_anomaly, device
):
    adapter_was_training = adapter.training
    prompt_was_training = (
        prompt_model.training if prompt_model is not None else False
    )
    adapter.eval()
    if prompt_model is not None:
        prompt_model.eval()
    labels_all, scores_all = [], []
    ratio_scores = {
        float(ratio): [] for ratio in args.source_validation_top_ratios
    }
    for batch in loader:
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        object_labels = (labels > 0).any(dim=1).float()
        features = encoder.encode_pointcloud(points, return_intermediate=True)
        tokens = select_multi_layer_tokens(
            features.get("patch_tokens", features["layer_feats"]),
            features["patch_idx"],
            args.feature_layers,
        )
        if prompt_model is not None:
            category_names = PointCloudDataset.PRESETS[args.dataset_name]
            object_names = [
                category_names[int(category)] for category in batch["category"]
            ]
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
        object_probability = pool_training_object_probability(
            args, global_logits, patch_logits
        )
        labels_all.extend(object_labels.detach().cpu().tolist())
        scores_all.extend(object_probability.detach().cpu().tolist())
        if args.object_pooling_mode != "legacy":
            patch_probabilities = torch.sigmoid(patch_logits.double())
            for ratio, collected in ratio_scores.items():
                candidate_probability = configurable_object_probability(
                    global_logits,
                    patch_probabilities,
                    args.object_pooling_mode,
                    ratio,
                    args.global_alpha,
                )
                collected.extend(candidate_probability.detach().cpu().tolist())
    metrics = safe_binary_metrics(labels_all, scores_all)
    adapter.train(adapter_was_training)
    if prompt_model is not None:
        prompt_model.train(prompt_was_training)
    result = {
        "val_object_auroc": float(metrics["auroc"]),
        "val_object_ap": float(metrics["ap"]),
        "val_samples": len(labels_all),
    }
    if args.object_pooling_mode != "legacy":
        ratio_metrics = {}
        for ratio, candidate_scores in ratio_scores.items():
            candidate_metrics = safe_binary_metrics(labels_all, candidate_scores)
            ratio_metrics[f"{ratio:g}"] = {
                "object_auroc": float(candidate_metrics["auroc"]),
                "object_ap": float(candidate_metrics["ap"]),
            }
        def source_ratio_key(ratio):
            item = ratio_metrics[f"{float(ratio):g}"]
            auroc = item["object_auroc"]
            average_precision = item["object_ap"]
            return (
                auroc if np.isfinite(auroc) else float("-inf"),
                average_precision
                if np.isfinite(average_precision)
                else float("-inf"),
                -args.source_validation_top_ratios.index(ratio),
            )

        selected_ratio = max(
            args.source_validation_top_ratios, key=source_ratio_key
        )
        result["source_top_ratio_metrics"] = ratio_metrics
        result["source_selected_top_ratio"] = float(selected_ratio)
    return result


def build_warmup_cosine_scheduler(optimizer, total_steps, warmup_ratio, minimum_lr):
    if total_steps <= 0:
        return None
    warmup_steps = int(total_steps * warmup_ratio)
    max_initial_lr = max(group["lr"] for group in optimizer.param_groups)
    minimum_factor = min(1.0, minimum_lr / max(max_initial_lr, 1e-12))

    def lr_factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return LambdaLR(optimizer, lr_factor)


def topk_mean_logits(patch_logits, top_percent):
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    count = max(1, int(patch_logits.shape[1] * top_percent))
    return patch_logits.topk(count, dim=1).values.mean(dim=1)


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


def checkpoint_visual_adapter_training_mode(checkpoint):
    """Read new metadata while treating every legacy checkpoint as fused-only."""
    return (
        checkpoint.get("visual_source_training_mode")
        or checkpoint.get("visual_adapter_training_mode")
        or "fused_only"
    )


def validate_visual_adapter_training_mode(args, checkpoint, checkpoint_path):
    mode = checkpoint_visual_adapter_training_mode(checkpoint)
    required = args.required_visual_adapter_training_mode
    if required != "any" and mode != required:
        raise RuntimeError(
            "Visual adapter training mode mismatch: "
            f"required={required}, checkpoint={mode}, path={checkpoint_path}"
        )
    args.visual_source_training_mode = mode


def build_static_prompt_model(args, encoder, adapter, device):
    if not args.use_static_prompt:
        return None

    prompt_checkpoint = None
    prompt_checkpoint_argument = args.resume_checkpoint or args.prompt_checkpoint
    if prompt_checkpoint_argument:
        prompt_path = Path(
            prompt_checkpoint_argument.format(train_category=args.train_category)
        )
        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"Prompt checkpoint not found: {prompt_path}"
            )
        prompt_checkpoint = torch.load(
            prompt_path, map_location=device, weights_only=False
        )
        if not checkpoint_uses_static_prompt(prompt_checkpoint):
            raise RuntimeError("Prompt checkpoint does not use static Prompts")
        if checkpoint_train_categories(prompt_checkpoint) != args.train_categories:
            raise RuntimeError("Prompt checkpoint train_categories mismatch")
        if list(prompt_checkpoint.get("feature_layers", [])) != list(
            args.feature_layers
        ):
            raise RuntimeError("Prompt checkpoint feature_layers mismatch")
        validate_visual_adapter_training_mode(
            args, prompt_checkpoint, prompt_path
        )
        validate_checkpoint_ddf3d_config(args, prompt_checkpoint, prompt_path)
        checkpoint_version = prompt_checkpoint.get(
            "static_prompt_version",
            prompt_checkpoint.get("geometric_mode_version"),
        )
        if checkpoint_version not in {
            args.static_prompt_version,
            LEGACY_STATIC_PROMPT_VERSION,
            LEGACY_NCRP_K1_VERSION,
        }:
            raise RuntimeError("Prompt checkpoint version mismatch")
        adapter.load_state_dict(prompt_checkpoint["adapter"])
    elif args.baseline_checkpoint:
        baseline_path = Path(
            args.baseline_checkpoint.format(train_category=args.train_category)
        )
        if not baseline_path.is_file():
            raise FileNotFoundError(
                f"Baseline checkpoint not found: {baseline_path}"
            )
        baseline = torch.load(
            baseline_path, map_location=device, weights_only=False
        )
        if checkpoint_train_categories(baseline) != args.train_categories:
            raise RuntimeError("Baseline checkpoint train_categories mismatch")
        if list(baseline.get("feature_layers", [])) != list(args.feature_layers):
            raise RuntimeError("Baseline checkpoint feature_layers mismatch")
        validate_visual_adapter_training_mode(args, baseline, baseline_path)
        validate_checkpoint_ddf3d_config(args, baseline, baseline_path)
        adapter.load_state_dict(baseline["adapter"])
    elif args.freeze_visual_adapter:
        raise ValueError(
            "baseline_checkpoint or prompt_checkpoint is required "
            "when freeze_visual_adapter=true"
        )

    if args.freeze_visual_adapter:
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad = False
    clip_model = encoder.open_clip_model
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    common_kwargs = {
        "clip_model": clip_model,
        "tokenizer": encoder.tokenizer,
        "num_prompts": args.num_abnormal_prompts,
        "num_normal_tokens": args.num_normal_tokens,
        "num_abnormal_tokens": args.num_abnormal_tokens,
        "prompt_template": args.prompt_template,
        "use_category_prompt": args.use_category_prompt,
    }
    if args.residual_prompt_enabled:
        model = NormalCenteredResidualPromptLearner(
            clip_model=clip_model,
            tokenizer=encoder.tokenizer,
            num_bases=args.residual_num_bases,
            num_normal_tokens=args.num_normal_tokens,
            prompt_template=args.prompt_template,
            use_category_prompt=args.use_category_prompt,
            gamma=args.residual_gamma,
            eps=args.residual_eps,
        ).to(device)
    else:
        model = StaticPromptLearner(**common_kwargs).to(device)
    if prompt_checkpoint is not None:
        model.load_state_dict(
            checkpoint_static_prompt_state(prompt_checkpoint), strict=True
        )
    return model

def main():
    args = parse_args()
    training_started = time.perf_counter()
    args.training_started = training_started
    args.training_time_offset = 0.0
    seed_report = set_seed(args.seed, args.num_workers)
    args.seed_report = seed_report
    train_categories, test_categories = resolve_categories(
        args.dataset_name, args.protocol, train_categories=args.train_categories
    )
    run_dir = Path(args.output_root) / args.train_category
    if (
        args.ddf3d_enabled
        and run_dir.exists()
        and any(run_dir.iterdir())
        and not args.resume_checkpoint
    ):
        raise FileExistsError(
            f"Refusing to overwrite an existing DDF-3D run directory: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train.log"
    if not args.resume_checkpoint or not train_log.is_file():
        train_log.write_text("", encoding="utf-8")
    else:
        log_line(train_log, "--- resumed training process ---")
    resolved = vars(args).copy()
    resolved.update(
        train_categories=train_categories,
        test_categories=test_categories,
        seed_report=seed_report,
    )
    write_yaml(run_dir / "config.yaml", resolved)
    log_line(train_log, f"Protocol: {args.protocol}")
    log_line(train_log, f"Train categories: {train_categories}")
    log_line(train_log, f"Test categories: {test_categories}")
    log_line(train_log, f"Seed report: {seed_report}")
    if args.use_static_prompt:
        log_line(
            train_log,
            f"Prompt learner={'NCRP-K1' if args.residual_prompt_enabled else 'Static'}",
        )
        if args.residual_prompt_enabled:
            log_line(
                train_log,
                "NCRP-K1 residual vectors=1"
                f" | gamma={args.residual_gamma:g}",
            )
        else:
            log_line(
                train_log,
                f"Learnable abnormal Prompts={args.num_abnormal_prompts}",
            )
        log_line(train_log, f"Prompt template: {args.prompt_template}")
        for category in train_categories:
            train_prompt_text = format_category_prompt(
                args.prompt_template, category, args.use_category_prompt
            )
            log_line(train_log, f"Train category prompt text [{category}]: {train_prompt_text}")
    else:
        log_line(
            train_log,
            "Stage 1: trainable multi-layer visual adapter"
            f" | mode={args.visual_adapter_training_mode}",
        )
    if args.ddf3d_enabled:
        log_line(
            train_log,
            "DDF-3D enabled"
            f" | fusion={args.ddf3d_fusion_mode}"
            f" | layers={args.ddf3d_layers}"
            f" | top_k={args.ddf3d_router_top_k}"
            f" | discrepancy={args.ddf3d_discrepancy_enabled}",
        )

    source_dataset = PointCloudDataset(
        args.data_root, split=args.train_split, classes=train_categories,
        dataset_name=args.dataset_name,
    )
    if len(source_dataset) == 0:
        raise FileNotFoundError("No source training samples")
    observed = assert_dataset_categories(source_dataset, train_categories, test_categories)
    log_line(train_log, f"Observed training path categories: {observed}")
    dataset = source_dataset
    max_samples, epochs = args.max_train_samples, args.epochs
    if args.debug:
        max_samples, epochs = max_samples or min(8, len(dataset)), 1
        log_line(train_log, f"Debug mode: epochs=1, max_train_samples={max_samples}")
    candidate_indices = list(range(len(dataset)))
    if max_samples > 0:
        candidate_indices = candidate_indices[:min(max_samples, len(candidate_indices))]
    train_indices, val_indices = stratified_holdout_indices(
        dataset, candidate_indices, args.validation_fraction, args.seed
    )
    training_data = Subset(dataset, train_indices)
    validation_data = Subset(dataset, val_indices) if val_indices else None
    log_line(train_log, f"Training samples: {len(train_indices)}")
    if validation_data is not None:
        log_line(train_log, f"Validation holdout samples: {len(val_indices)}")
    train_seed_kwargs = dataloader_seed_kwargs(args.seed)
    loader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.device == "cuda",
        **train_seed_kwargs,
    )
    val_loader = (
        DataLoader(
            validation_data, batch_size=args.batch_size, shuffle=False, drop_last=False,
            num_workers=args.num_workers, pin_memory=args.device == "cuda",
            **dataloader_seed_kwargs(args.seed),
        )
        if validation_data is not None else None
    )

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    encoder = ULIP2Encoder(
        args.model_path, device=device, num_points=args.num_points,
        return_layers=tuple(args.return_layers),
        return_clip=args.use_static_prompt,
    )
    log_line(
        train_log,
        "PointBERT blocks="
        f"{encoder.pointbert_block_count} | configured layer -> Python index="
        f"{encoder.layer_block_indices}",
    )
    adapter = build_patch_adapter(args).to(device)
    parameter_counts = None
    prompt_model = build_static_prompt_model(args, encoder, adapter, device)
    if prompt_model is not None:
        log_line(
            train_log,
            "Loaded visual adapter training mode="
            f"{getattr(args, 'visual_source_training_mode', 'unknown')}",
        )
        parameter_counts = prompt_parameter_counts(prompt_model)
        log_line(train_log, f"Prompt parameter counts: {parameter_counts}")
        parameter_groups = [
            {
                "params": prompt_model.parameters(),
                "lr": args.prompt_learning_rate,
            },
        ]
        if not args.freeze_visual_adapter:
            parameter_groups.append({
                "params": adapter.parameters(), "lr": args.learning_rate
            })
        optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)
        fixed_normal = fixed_anomaly = None
    else:
        optimizer = AdamW(
            adapter.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        fixed_normal = encoder.encode_text_templates(
            args.normal_templates
        ).detach()
        fixed_anomaly = encoder.encode_text_templates(
            args.anomaly_templates
        ).detach()

    scheduler = build_warmup_cosine_scheduler(
        optimizer, epochs * len(loader), args.warmup_ratio, args.minimum_learning_rate
    )
    start_epoch = 0
    resume_payload = None
    if args.resume_checkpoint:
        resume_path = Path(
            args.resume_checkpoint.format(train_category=args.train_category)
        )
        resume_payload = torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if "adapter" not in resume_payload:
            raise RuntimeError(f"Resume checkpoint lacks adapter state: {resume_path}")
        adapter.load_state_dict(resume_payload["adapter"], strict=True)
        if "optimizer" not in resume_payload or "scheduler" not in resume_payload:
            raise RuntimeError(
                "Resume checkpoint lacks optimizer/scheduler state: "
                f"{resume_path}"
            )
        optimizer.load_state_dict(resume_payload["optimizer"])
        if scheduler is not None and resume_payload["scheduler"] is not None:
            scheduler.load_state_dict(resume_payload["scheduler"])
        if "dataloader_generator_state" in resume_payload:
            train_seed_kwargs["generator"].set_state(
                resume_payload["dataloader_generator_state"]
            )
        start_epoch = int(resume_payload.get("epoch", 0))
        args.training_time_offset = float(
            resume_payload.get("training_time_seconds") or 0.0
        )
        if start_epoch < 0 or start_epoch >= epochs:
            raise RuntimeError(
                f"Resume epoch {start_epoch} is outside [0, {epochs})"
            )
        log_line(
            train_log,
            f"Resume NCRP training: checkpoint={resume_path} start_epoch={start_epoch}",
        )
    focal_loss = BinaryFocalLoss(args.focal_gamma, args.focal_alpha)
    dice_loss = BinaryDiceLoss()
    object_loss = nn.BCELoss()
    best_loss = float(
        resume_payload.get("best_loss_so_far", float("inf"))
        if resume_payload is not None
        else float("inf")
    )
    best_val_object_auroc = float(
        resume_payload.get("best_val_object_auroc_so_far", float("-inf"))
        if resume_payload is not None
        else float("-inf")
    )
    final_training_statistics = None
    last_validation_metrics = None
    ncrp_training_curve = list(
        (((resume_payload or {}).get("training_statistics") or {}).get("ncrp") or {}).get(
            "training_curve", []
        )
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, epochs):
        adapter.train(
            not args.use_static_prompt or not args.freeze_visual_adapter
        )
        if prompt_model is not None:
            prompt_model.train()
        totals = {
            name: 0.0
            for name in (
                "local",
                "object",
                "prompt_diversity",
                "focal",
                "dice",
                "route_balance",
                "total",
            )
        }
        routing_statistics = (
            RoutingStatistics(adapter.layers)
            if isinstance(adapter, DDF3DAdapter)
            else None
        )
        ncrp_normal_residual_sum = 0.0
        ncrp_abnormal_residual_sum = 0.0
        ncrp_normal_residual_count = 0
        ncrp_abnormal_residual_count = 0
        for batch in loader:
            ddf3d_score_output = None
            points = batch["points"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            category_names = PointCloudDataset.PRESETS[args.dataset_name]
            object_names = [
                category_names[int(category)] for category in batch["category"]
            ]
            object_labels = (labels > 0).any(dim=1).float()
            features = encoder.encode_pointcloud(
                points, return_intermediate=True
            )
            tokens = select_multi_layer_tokens(
                features.get("patch_tokens", features["layer_feats"]),
                features["patch_idx"],
                args.feature_layers,
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
                diversity = patch_embeddings.sum() * 0.0
                if args.use_prompt_diversity_loss:
                    diversity = prompt_diversity_loss(
                        score_output["diversity_embeddings"]
                    )
                residual_masks = None
                if args.residual_prompt_enabled:
                    anomaly_mask, normal_mask, _, _ = point_mask_to_patch_targets(
                        labels,
                        features["patch_idx"],
                        args.patch_anomaly_threshold,
                    )
                    residual_masks = (anomaly_mask, normal_mask)
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
                        patch_embeddings,
                        fixed_normal,
                        fixed_anomaly,
                        args.temperature,
                    )
                    global_logits = patch_text_logits(
                        features["concat"].unsqueeze(1),
                        fixed_normal,
                        fixed_anomaly,
                        args.temperature,
                    ).squeeze(1)
                diversity = patch_embeddings.sum() * 0.0
                residual_masks = None

            if isinstance(adapter, DDF3DAdapter):
                ddf3d_score_output = score_output.get("ddf3d", score_output)
                routing_statistics.update(ddf3d_score_output["routing_weights"])

            point_logits = patch_to_point(
                patch_logits, features["patch_idx"], labels.shape[1]
            )
            object_probability = pool_training_object_probability(
                args, global_logits, patch_logits
            )
            focal_value = args.focal_weight * focal_loss(point_logits, labels)
            dice_value = args.dice_weight * dice_loss(point_logits, labels)
            local = focal_value + dice_value
            object_value = object_loss(
                object_probability, object_labels.to(object_probability.dtype)
            )
            total = local + args.object_weight * object_value
            route_balance = patch_logits.sum() * 0.0
            if ddf3d_score_output is not None and any(
                parameter.requires_grad for parameter in adapter.parameters()
            ):
                route_balance = adapter.route_balance_loss(
                    ddf3d_score_output["routing_weights"]
                )
                total = total + args.ddf3d_route_balance_weight * route_balance
            if prompt_model is not None and args.use_prompt_diversity_loss:
                total = total + args.lambda_prompt_diversity * diversity
            optimizer.zero_grad(set_to_none=True)
            if total.requires_grad:
                total.backward()
                if args.gradient_clip_norm > 0:
                    trainable_parameters = [
                        parameter
                        for group in optimizer.param_groups
                        for parameter in group["params"]
                        if parameter.grad is not None
                    ]
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, args.gradient_clip_norm
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            for name, value in (
                ("local", local),
                ("object", object_value),
                ("prompt_diversity", diversity),
                ("focal", focal_value),
                ("dice", dice_value),
                ("route_balance", route_balance),
                ("total", total),
            ):
                totals[name] += float(value.detach())
            if args.residual_prompt_enabled and residual_masks is not None:
                anomaly_mask, normal_mask = residual_masks
                residual_norms = score_output["residual_norms"].detach()
                if normal_mask.any():
                    ncrp_normal_residual_sum += float(residual_norms[normal_mask].sum())
                    ncrp_normal_residual_count += int(normal_mask.sum())
                if anomaly_mask.any():
                    ncrp_abnormal_residual_sum += float(residual_norms[anomaly_mask].sum())
                    ncrp_abnormal_residual_count += int(anomaly_mask.sum())

        averages = {
            name: value / max(1, len(loader))
            for name, value in totals.items()
        }
        display_names = ["local", "object"]
        if prompt_model is not None and args.use_prompt_diversity_loss:
            display_names.append("prompt_diversity")
        if prompt_model is not None and args.residual_prompt_enabled:
            display_names = ["focal", "dice", "object"]
        if isinstance(adapter, DDF3DAdapter):
            display_names.append("route_balance")
        display_names.append("total")
        epoch_message = f"Epoch {epoch + 1}/{epochs} | " + " | ".join(
            f"{name}={averages[name]:.6f}" for name in display_names
        )
        epoch_message += f" | lr={optimizer.param_groups[0]['lr']:.8f}"
        if prompt_model is not None and not args.residual_prompt_enabled:
            epoch_message += (
                f" | static_abnormal_prompts={args.num_abnormal_prompts}"
            )
        ncrp_normal_residual = ncrp_normal_residual_sum / max(
            1, ncrp_normal_residual_count
        )
        ncrp_abnormal_residual = ncrp_abnormal_residual_sum / max(
            1, ncrp_abnormal_residual_count
        )
        # Stage 1 has no Prompt model and therefore never creates score_output.
        # Keep the shared epoch diagnostics path safe for visual-only training.
        ncrp_score_output = (
            score_output
            if args.residual_prompt_enabled and prompt_model is not None
            else {}
        )
        ncrp_directions = ncrp_score_output.get("local_projected_directions")
        if ncrp_directions is None:
            ncrp_directions = ncrp_score_output.get("projected_directions")
        ncrp_gram = (
            (
                ncrp_directions.detach()
                @ ncrp_directions.detach().transpose(-1, -2)
            ).mean(0)
            if ncrp_directions is not None
            else torch.eye(args.residual_num_bases, device=device)
        )
        basis_normal_inner = ncrp_score_output.get(
            "basis_normal_max_abs_inner_product"
        )
        ncrp_epoch = {
            "epoch": epoch + 1,
            "total_loss": averages["total"],
            "focal_loss": averages["focal"],
            "dice_loss": averages["dice"],
            "object_loss": averages["object"],
            "basis_gram_off_diagonal_mean": 0.0,
            "basis_usage": [1.0],
            "assignment_entropy": None,
            "normalized_assignment_entropy": None,
            "assignment_diagnostics_status": "not_applicable_single_basis",
            "basis_usage_status": "structural_single_basis",
            "max_assignment_weight": 1.0,
            "normal_residual_norm": ncrp_normal_residual,
            "abnormal_residual_norm": ncrp_abnormal_residual,
            "basis_normal_max_abs_inner_product": (
                float(basis_normal_inner.detach().max())
                if basis_normal_inner is not None
                else None
            ),
        }
        if args.residual_prompt_enabled:
            ncrp_training_curve.append(ncrp_epoch)
        if isinstance(adapter, DDF3DAdapter):
            if adapter.settings.fusion_mode == "global_softmax":
                routing_result = global_routing_weights(
                    adapter.layers, adapter.layer_weights()
                )
            else:
                routing_result = routing_statistics.result()
            write_json(
                run_dir
                / "routing_stats"
                / args.train_category
                / "train.json",
                routing_result,
            )
            layer_weight_values = (
                list(routing_result.get("mean_weight", {}).values())
                if "mean_weight" in routing_result
                else list(routing_result.values())
                if adapter.settings.fusion_mode == "global_softmax"
                else None
            )
        else:
            routing_result = None
            layer_weight_values = [
                float(value) for value in adapter.layer_weights().detach().cpu()
            ]
        final_training_statistics = {
            "visual_adapter_training_mode": args.visual_adapter_training_mode,
            "visual_adapter_layer_weights": layer_weight_values,
            "ddf3d_routing": routing_result,
            "ncrp": {
                "enabled": bool(args.residual_prompt_enabled),
                **ncrp_epoch,
                "basis_gram_matrix": ncrp_gram.detach().cpu().tolist(),
                "training_curve": list(ncrp_training_curve),
            },
        }
        if prompt_model is not None and args.residual_prompt_enabled:
            epoch_message += (
                " | basis_usage=[1.0]"
                " | assignment=not_applicable_single_basis"
                " | normal_residual_norm="
                f"{ncrp_normal_residual:.6f}"
                " | abnormal_residual_norm="
                f"{ncrp_abnormal_residual:.6f}"
            )
        log_line(train_log, epoch_message)

        validation_metrics = None
        if val_loader is not None and (
            (epoch + 1) % args.validation_interval == 0
            or epoch + 1 == epochs
        ):
            validation_metrics = evaluate_holdout_object_metrics(
                args,
                val_loader,
                encoder,
                adapter,
                prompt_model,
                fixed_normal,
                fixed_anomaly,
                device,
            )
            last_validation_metrics = validation_metrics
            log_line(
                train_log,
                "Validation"
                f" | object_auroc={validation_metrics['val_object_auroc']:.6f}"
                f" | object_ap={validation_metrics['val_object_ap']:.6f}"
                f" | samples={validation_metrics['val_samples']}",
            )
            if "source_top_ratio_metrics" in validation_metrics:
                log_line(
                    train_log,
                    "Source-only top-ratio validation"
                    f" | metrics={validation_metrics['source_top_ratio_metrics']}"
                    " | selected="
                    f"{validation_metrics['source_selected_top_ratio']:g}",
                )

        if averages["total"] < best_loss:
            best_loss = averages["total"]
            state = build_checkpoint_state(
                args,
                adapter,
                prompt_model,
                epoch + 1,
                best_loss,
                "loss",
                best_loss,
            )
            state["prompt_parameter_counts"] = parameter_counts
            state["training_statistics"] = final_training_statistics
            state["optimizer"] = optimizer.state_dict()
            state["scheduler"] = scheduler.state_dict() if scheduler is not None else None
            state["best_loss_so_far"] = best_loss
            state["best_val_object_auroc_so_far"] = best_val_object_auroc
            state["dataloader_generator_state"] = train_seed_kwargs[
                "generator"
            ].get_state()
            torch.save(state, run_dir / "best_loss.pth")
            if args.checkpoint_metric == "loss":
                torch.save(state, run_dir / "best.pth")
        if validation_metrics is not None:
            val_object_auroc = validation_metrics['val_object_auroc']
            if (
                np.isfinite(val_object_auroc)
                and val_object_auroc > best_val_object_auroc
            ):
                best_val_object_auroc = val_object_auroc
                state = build_checkpoint_state(
                    args,
                    adapter,
                    prompt_model,
                    epoch + 1,
                    averages["total"],
                    "val_object_auroc",
                    best_val_object_auroc,
                )
                state.update(validation_metrics)
                state["prompt_parameter_counts"] = parameter_counts
                state["training_statistics"] = final_training_statistics
                state["optimizer"] = optimizer.state_dict()
                state["scheduler"] = scheduler.state_dict() if scheduler is not None else None
                state["best_loss_so_far"] = best_loss
                state["best_val_object_auroc_so_far"] = best_val_object_auroc
                state["dataloader_generator_state"] = train_seed_kwargs[
                    "generator"
                ].get_state()
                torch.save(state, run_dir / "best_object_auroc.pth")
                if args.checkpoint_metric == "val_object_auroc":
                    torch.save(state, run_dir / "best.pth")
        if args.residual_prompt_enabled:
            last_state = build_checkpoint_state(
                args,
                adapter,
                prompt_model,
                epoch + 1,
                averages["total"],
                "last_epoch",
                averages["total"],
            )
            last_state["prompt_parameter_counts"] = parameter_counts
            last_state["training_statistics"] = final_training_statistics
            last_state["optimizer"] = optimizer.state_dict()
            last_state["scheduler"] = scheduler.state_dict() if scheduler is not None else None
            last_state["best_loss_so_far"] = best_loss
            last_state["best_val_object_auroc_so_far"] = best_val_object_auroc
            last_state["dataloader_generator_state"] = train_seed_kwargs[
                "generator"
            ].get_state()
            torch.save(last_state, run_dir / "last.pth")
    if args.checkpoint_metric == "val_object_auroc" and best_val_object_auroc == float("-inf"):
        raise RuntimeError("No finite validation object AUROC was computed")
    log_line(train_log, f"Best loss checkpoint: {run_dir / 'best_loss.pth'} | loss={best_loss:.6f}")
    if best_val_object_auroc > float("-inf"):
        log_line(
            train_log,
            f"Best object-AUROC checkpoint: {run_dir / 'best_object_auroc.pth'}"
            f" | val_object_auroc={best_val_object_auroc:.6f}",
        )
    log_line(train_log, f"Selected checkpoint: {run_dir / 'best.pth'} | metric={args.checkpoint_metric}")
    training_time_seconds = (
        args.training_time_offset + time.perf_counter() - training_started
    )
    log_line(
        train_log,
        "Training resources"
        f" | time_seconds={training_time_seconds:.3f}"
        " | peak_gpu_memory_bytes="
        f"{int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0}",
    )
    write_yaml(
        run_dir / "training_complete.yaml",
        {
            "train_category": args.train_category,
            "train_categories": list(args.train_categories),
            "dataset_name": args.dataset_name,
            "epochs": epochs,
            "best_loss": float(best_loss),
            "best_val_object_auroc": (
                float(best_val_object_auroc)
                if best_val_object_auroc > float("-inf") else None
            ),
            "checkpoint_metric": args.checkpoint_metric,
            "seed": args.seed,
            "seed_report": seed_report,
            "validation_fraction": args.validation_fraction,
            "validation_interval": args.validation_interval,
            "use_static_prompt": args.use_static_prompt,
            "static_prompt_version": (
                args.static_prompt_version if args.use_static_prompt else None
            ),
            "num_abnormal_prompts": (
                args.num_abnormal_prompts if args.use_static_prompt else None
            ),
            "prompt_template": (
                args.prompt_template if args.use_static_prompt else None
            ),
            "prompt_score_temperature": (
                args.prompt_score_temperature
                if args.use_static_prompt else None
            ),
            "object_pooling_mode": args.object_pooling_mode,
            "object_top_ratio": args.object_top_ratio,
            "visual_adapter_training_mode": args.visual_adapter_training_mode,
            "required_visual_adapter_training_mode": (
                args.required_visual_adapter_training_mode
            ),
            "visual_source_training_mode": getattr(
                args, "visual_source_training_mode", None
            ),
            "residual_prompt_enabled": args.residual_prompt_enabled,
            "residual_num_bases": args.residual_num_bases,
            "residual_gamma": args.residual_gamma,
            "residual_eps": args.residual_eps,
            "ddf3d_enabled": bool(args.ddf3d_enabled),
            "ddf3d_fusion_mode": (
                args.ddf3d_fusion_mode if args.ddf3d_enabled else None
            ),
            "ddf3d_layers": (
                list(args.ddf3d_layers) if args.ddf3d_enabled else None
            ),
            "prompt_parameter_counts": parameter_counts,
            "training_statistics": final_training_statistics,
            "training_time_seconds": training_time_seconds,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
            ),
            "source_top_ratio_metrics": (
                last_validation_metrics.get("source_top_ratio_metrics")
                if last_validation_metrics is not None
                else None
            ),
            "source_selected_top_ratio": (
                last_validation_metrics.get("source_selected_top_ratio")
                if last_validation_metrics is not None
                else None
            ),
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
