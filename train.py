"""Train the linear baseline or prompt-only 3D geometric CAP/DAP."""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from data.anomaly_datasets import PointCloudDataset
from loss import BinaryDiceLoss, BinaryFocalLoss
from models.geometric_cap import (
    GeometricCompoundPromptLearner,
    abnormal_prompt_orthogonal_loss,
    encode_geometric_prompts,
    encode_prior_enabled_abnormal_prompts,
    geometric_anomaly_logits,
)
from models.geometric_dap import (
    PointCloudAbnormalityPrior,
    geometric_prior_invariance_loss,
    random_se3_transform,
    normal_sample_prior_loss,
)
from models.geometric_mode_graph import random_se3_transform as random_mode_se3_transform
from models.geometric_mode_prompt import (
    GeometricModePromptLearner,
    encode_geometric_mode_prompts,
    format_category_prompt,
    geometry_gate_supervision_loss,
    mode_aware_anomaly_logits,
    mode_diversity_loss,
    mode_entropy_regularization,
    normalized_mode_entropy,
    point_mask_to_patch_targets,
    residual_suppression_loss,
    se3_sanity_loss,
    sinkhorn_mode_assignment_loss,
)
from models.trainable_baseline import (
    MultiLayerPatchAdapter,
    patch_text_logits,
    patch_to_point,
    aggregate_object_probability,
    select_multi_layer_tokens,
)
from models.ulip2_encoder import ULIP2Encoder
from one_rest_protocol import (
    assert_dataset_categories,
    load_yaml,
    log_line,
    resolve_categories,
    validate_one_rest_flags,
    write_yaml,
)


DEFAULT_CONFIG = "configs/trainable_baseline.yaml"


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y"}:
        return True
    if value.lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean, got {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train ULIP patch baseline or 3D-CAP/DAP.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", choices=("baseline", "one_rest"), default="baseline")
    parser.add_argument("--train_category")
    parser.add_argument("--train_class", dest="train_category")
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
    parser.add_argument("--prior_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--minimum_learning_rate", type=float, default=1e-6)
    parser.add_argument("--gradient_clip_norm", type=float, default=0.0)
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
    parser.add_argument("--use_geometric_cap", type=str2bool, default=False)
    parser.add_argument("--use_geometric_dap", type=str2bool, default=False)
    parser.add_argument("--num_geometric_abnormal_prompts", type=int, default=10)
    parser.add_argument("--num_normal_tokens", type=int, default=4)
    parser.add_argument("--num_abnormal_tokens", type=int, default=4)
    parser.add_argument("--geometric_prompt_suffix", default="a point cloud patch of {article} {category}")
    parser.add_argument("--top_m_abnormal_patches", type=int, default=10)
    parser.add_argument("--prior_hidden_dim", type=int, default=512)
    parser.add_argument("--patch_geo_desc_dim", type=int, default=4)
    parser.add_argument("--geometry_graph_dim", type=int, default=128)
    parser.add_argument("--geometry_graph_k", type=int, default=8)
    parser.add_argument("--geometry_graph_layers", type=int, default=2)
    parser.add_argument("--geometry_routing_temperature", type=float, default=0.2)
    parser.add_argument("--use_geometric_mode_prompt", type=str2bool, default=False)
    parser.add_argument("--use_category_prompt", type=str2bool, default=True)
    parser.add_argument("--prompt_template", default="a point cloud patch of a {category}")
    parser.add_argument("--num_geometric_modes", type=int, default=10)
    parser.add_argument("--mode_router_temperature", type=float, default=0.2)
    parser.add_argument("--mode_score_type", choices=("weighted_sum", "logsumexp", "max"), default="weighted_sum")
    parser.add_argument("--mode_score_temperature", type=float, default=0.1)
    parser.add_argument("--mode_residual_scale", type=float, default=0.1)
    parser.add_argument("--mode_prompt_hidden_dim", type=int, default=512)
    parser.add_argument("--use_mode_specific_residual", type=str2bool, default=True)
    parser.add_argument("--use_mode_weighted_scoring", type=str2bool, default=True)
    parser.add_argument("--use_patch_mode_routing", type=str2bool, default=True)
    parser.add_argument("--mode_anomaly_patch_threshold", type=float, default=0.05)
    parser.add_argument("--geometric_mode_version", default="se3_mode_prompt_v7_gate_sinkhorn")
    parser.add_argument("--use_geometry_abnormal_gate", type=str2bool, default=False)
    parser.add_argument("--geometry_gate_logit_scale", type=float, default=1.0)
    parser.add_argument("--lambda_geometry_gate", type=float, default=0.2)
    parser.add_argument("--geometry_gate_margin", type=float, default=0.5)
    parser.add_argument("--use_sinkhorn_mode_assignment", type=str2bool, default=False)
    parser.add_argument("--lambda_mode_assignment", type=float, default=0.05)
    parser.add_argument("--sinkhorn_epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iterations", type=int, default=3)
    parser.add_argument("--use_mode_diversity_loss", type=str2bool, default=True)
    parser.add_argument("--use_mode_entropy_loss", type=str2bool, default=True)
    parser.add_argument("--use_residual_suppression_loss", type=str2bool, default=True)
    parser.add_argument("--lambda_mode_diversity", type=float, default=0.05)
    parser.add_argument("--lambda_mode_entropy", type=float, default=0.05)
    parser.add_argument("--mode_conditional_entropy_weight", type=float, default=0.5)
    parser.add_argument("--lambda_residual_suppression", type=float, default=0.1)
    parser.add_argument("--lambda_se3_sanity", type=float, default=0.0)
    parser.add_argument("--lambda_geometry_invariance", type=float, default=0.1)
    parser.add_argument("--lambda_prompt_orthogonal", type=float, default=0.1)
    parser.add_argument("--lambda_prior", type=float, default=0.1)
    parser.add_argument("--freeze_plm", type=str2bool, default=True)
    parser.add_argument("--freeze_text_encoder", type=str2bool, default=True)
    parser.add_argument("--freeze_visual_adapter", type=str2bool, default=True)
    parser.add_argument("--baseline_checkpoint")
    return parser


def parse_args():
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--config", default=DEFAULT_CONFIG)
    config_args, _ = first.parse_known_args()
    config = load_yaml(config_args.config)
    parser = build_parser()
    valid = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - valid)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    parser.set_defaults(**config)
    args = parser.parse_args()
    if not args.train_category:
        parser.error("--train_category is required")
    missing_layers = [layer for layer in args.feature_layers if layer not in args.return_layers]
    if missing_layers:
        parser.error(f"feature_layers must be included in return_layers: {missing_layers}")
    if args.use_geometric_dap and not args.use_geometric_cap:
        parser.error("use_geometric_dap requires use_geometric_cap=true")
    if args.use_geometric_mode_prompt and (args.use_geometric_cap or args.use_geometric_dap):
        parser.error("geometric mode prompting is independent of the legacy CAP/DAP path")
    if args.use_geometric_mode_prompt and not (args.freeze_plm and args.freeze_text_encoder):
        parser.error("geometric mode prompting requires frozen PLM and text encoder")
    if args.use_geometric_cap and not (args.freeze_plm and args.freeze_text_encoder):
        parser.error("3D-CAP requires freeze_plm=true and freeze_text_encoder=true")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("warmup_ratio must be in [0, 1)")
    if args.minimum_learning_rate < 0 or args.gradient_clip_norm < 0:
        parser.error("minimum_learning_rate and gradient_clip_norm must be non-negative")
    if args.use_sinkhorn_mode_assignment and not args.use_geometric_mode_prompt:
        parser.error("Sinkhorn mode assignment requires geometric mode prompting")
    validate_one_rest_flags(args)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



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


def build_cap_modules(args, encoder, adapter, device):
    if not args.use_geometric_cap:
        return None, None
    if args.freeze_visual_adapter:
        if not args.baseline_checkpoint:
            raise ValueError(
                "baseline_checkpoint is required when freeze_visual_adapter=true"
            )
        baseline_path = Path(
            args.baseline_checkpoint.format(train_category=args.train_category)
        )
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
        baseline = torch.load(
            baseline_path, map_location=device, weights_only=False
        )
        if baseline.get("train_category") != args.train_category:
            raise RuntimeError("Baseline checkpoint train_category mismatch")
        if "feature_layers" not in baseline:
            raise RuntimeError(
                "Baseline checkpoint uses the old single-layer adapter; retrain it with feature_layers=[2,5,8,11]"
            )
        if list(baseline["feature_layers"]) != list(args.feature_layers):
            raise RuntimeError(
                "Baseline checkpoint feature_layers do not match the current config"
            )
        adapter.load_state_dict(baseline["adapter"])
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad = False

    clip_model = encoder.open_clip_model
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    prompt_learner = GeometricCompoundPromptLearner(
        clip_model,
        encoder.tokenizer,
        args.num_geometric_abnormal_prompts,
        args.num_normal_tokens,
        args.num_abnormal_tokens,
        args.geometric_prompt_suffix,
    ).to(device)
    prior_network = None
    if args.use_geometric_dap:
        prior_network = PointCloudAbnormalityPrior(
            feature_dim=args.text_dim,
            text_dim=args.text_dim,
            prompt_token_dim=prompt_learner.token_width,
            graph_dim=args.geometry_graph_dim,
            hidden_dim=args.prior_hidden_dim,
            graph_k=args.geometry_graph_k,
            graph_layers=args.geometry_graph_layers,
            routing_temperature=args.geometry_routing_temperature,
            top_m=args.top_m_abnormal_patches,
        ).to(device)
    return prompt_learner, prior_network


def build_geometric_mode_model(args, encoder, adapter, device):
    if not args.use_geometric_mode_prompt:
        return None
    if args.freeze_visual_adapter:
        if not args.baseline_checkpoint:
            raise ValueError("baseline_checkpoint is required when freeze_visual_adapter=true")
        baseline_path = Path(args.baseline_checkpoint.format(train_category=args.train_category))
        if not baseline_path.is_file():
            raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_path}")
        baseline = torch.load(baseline_path, map_location=device, weights_only=False)
        if baseline.get("train_category") != args.train_category:
            raise RuntimeError("Baseline checkpoint train_category mismatch")
        if list(baseline.get("feature_layers", [])) != list(args.feature_layers):
            raise RuntimeError("Baseline checkpoint feature_layers mismatch")
        adapter.load_state_dict(baseline["adapter"])
        adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad = False
    clip_model = encoder.open_clip_model
    clip_model.eval()
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    return GeometricModePromptLearner(
        clip_model=clip_model,
        tokenizer=encoder.tokenizer,
        num_modes=args.num_geometric_modes,
        num_normal_tokens=args.num_normal_tokens,
        num_abnormal_tokens=args.num_abnormal_tokens,
        prompt_template=args.prompt_template,
        use_category_prompt=args.use_category_prompt,
        graph_dim=args.geometry_graph_dim,
        graph_k=args.geometry_graph_k,
        graph_layers=args.geometry_graph_layers,
        router_temperature=args.mode_router_temperature,
        residual_scale=args.mode_residual_scale,
        modulator_hidden_dim=args.mode_prompt_hidden_dim,
        use_mode_specific_residual=args.use_mode_specific_residual,
    ).to(device)


def main():
    args = parse_args()
    set_seed(args.seed)
    train_categories, test_categories = resolve_categories(
        args.dataset_name, args.protocol, args.train_category
    )
    run_dir = Path(args.output_root) / args.train_category
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train.log"
    train_log.write_text("", encoding="utf-8")
    resolved = vars(args).copy()
    resolved.update(train_categories=train_categories, test_categories=test_categories)
    write_yaml(run_dir / "config.yaml", resolved)
    log_line(train_log, f"Protocol: {args.protocol}")
    log_line(train_log, f"Train categories: {train_categories}")
    log_line(train_log, f"Test categories: {test_categories}")
    if args.use_geometric_mode_prompt:
        log_line(train_log, "GeometricModePrompt=True")
        train_prompt_text = format_category_prompt(
            args.prompt_template, args.train_category, args.use_category_prompt
        )
        log_line(train_log, f"Prompt template: {args.prompt_template}")
        log_line(train_log, f"Train category prompt text: {train_prompt_text}")
    else:
        log_line(
            train_log,
            f"Geometric CAP={args.use_geometric_cap}, DAP={args.use_geometric_dap}",
        )

    dataset = PointCloudDataset(
        args.data_root, split=args.train_split, classes=train_categories,
        dataset_name=args.dataset_name,
    )
    if len(dataset) == 0:
        raise FileNotFoundError("No source training samples")
    observed = assert_dataset_categories(dataset, train_categories, test_categories)
    log_line(train_log, f"Observed training path categories: {observed}")
    max_samples, epochs = args.max_train_samples, args.epochs
    if args.debug:
        max_samples, epochs = max_samples or min(8, len(dataset)), 1
        log_line(train_log, f"Debug mode: epochs=1, max_train_samples={max_samples}")
    training_data = dataset if max_samples <= 0 else Subset(
        dataset, range(min(max_samples, len(dataset)))
    )
    loader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=args.device == "cuda",
    )

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    encoder = ULIP2Encoder(
        args.model_path, device=device, num_points=args.num_points,
        return_layers=tuple(args.return_layers),
        return_clip=args.use_geometric_cap or args.use_geometric_mode_prompt,
    )
    adapter = MultiLayerPatchAdapter(
        args.token_dim, args.text_dim, len(args.feature_layers)
    ).to(device)
    prompt_learner, prior_network = build_cap_modules(args, encoder, adapter, device)
    mode_prompt_model = build_geometric_mode_model(args, encoder, adapter, device)
    if mode_prompt_model is not None:
        parameter_groups = [
            {"params": mode_prompt_model.prompt_bank.parameters(), "lr": args.prompt_learning_rate},
            {"params": mode_prompt_model.graph_encoder.parameters(), "lr": args.prior_learning_rate},
            {"params": mode_prompt_model.mode_router.parameters(), "lr": args.prior_learning_rate},
            {"params": mode_prompt_model.prompt_modulator.parameters(), "lr": args.prior_learning_rate},
        ]
        if not args.freeze_visual_adapter:
            parameter_groups.append({"params": adapter.parameters(), "lr": args.learning_rate})
        optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)
        fixed_normal = fixed_anomaly = None
    elif args.use_geometric_cap:
        parameter_groups = [{"params": prompt_learner.parameters(), "lr": args.prompt_learning_rate}]
        if not args.freeze_visual_adapter:
            parameter_groups.append(
                {"params": adapter.parameters(), "lr": args.learning_rate}
            )
        if prior_network is not None:
            parameter_groups.append({"params": prior_network.parameters(), "lr": args.prior_learning_rate})
        optimizer = AdamW(parameter_groups, weight_decay=args.weight_decay)
        fixed_normal = fixed_anomaly = None
    else:
        optimizer = AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        fixed_normal = encoder.encode_text_templates(args.normal_templates).detach()
        fixed_anomaly = encoder.encode_text_templates(args.anomaly_templates).detach()

    scheduler = build_warmup_cosine_scheduler(
        optimizer, epochs * len(loader), args.warmup_ratio, args.minimum_learning_rate
    )
    focal_loss = BinaryFocalLoss(args.focal_gamma, args.focal_alpha)
    dice_loss = BinaryDiceLoss()
    object_loss = nn.BCELoss()
    best_loss = float("inf")
    for epoch in range(epochs):
        adapter.train(
            not (args.use_geometric_cap or args.use_geometric_mode_prompt)
            or not args.freeze_visual_adapter
        )
        if prompt_learner is not None:
            prompt_learner.train()
        if prior_network is not None:
            prior_network.train()
        if mode_prompt_model is not None:
            mode_prompt_model.train()
        totals = {name: 0.0 for name in (
            "local", "object", "orthogonal", "prior", "invariance",
            "mode_diversity", "mode_entropy", "mode_assignment",
            "geometry_gate", "residual_suppression", "se3_sanity", "total",
        )}
        mode_usage_sum = (
            torch.zeros(args.num_geometric_modes, device=device)
            if mode_prompt_model is not None else None
        )
        mode_usage_sq_sum = (
            torch.zeros(args.num_geometric_modes, device=device)
            if mode_prompt_model is not None else None
        )
        mode_sample_count = 0
        mode_max_sum = mode_entropy_sum = delta_norm_sum = 0.0
        normal_delta_sum = abnormal_delta_sum = 0.0
        normal_delta_count = abnormal_delta_count = 0
        anomaly_patch_usage_sum = torch.zeros(args.num_geometric_modes, device=device)
        normal_patch_usage_sum = torch.zeros(args.num_geometric_modes, device=device)
        anomaly_patch_count = normal_patch_count = 0
        anomaly_patch_entropy_sum = normal_patch_entropy_sum = 0.0
        anomaly_patch_confidence_sum = normal_patch_confidence_sum = 0.0
        anomaly_gate_probability_sum = normal_gate_probability_sum = 0.0
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            object_names = [args.train_category] * points.shape[0]
            object_labels = (labels > 0).any(dim=1).float()
            features = encoder.encode_pointcloud(points, return_intermediate=True)
            tokens = select_multi_layer_tokens(
                features["layer_feats"], features["patch_idx"], args.feature_layers
            )
            patch_embeddings = adapter(tokens)
            zero = patch_embeddings.sum() * 0.0
            orthogonal = prior_loss = invariance_loss = zero
            diversity = entropy_loss = assignment_loss = gate_loss = suppression = sanity = zero

            if mode_prompt_model is not None:
                geometry = mode_prompt_model.forward_geometry(points, features["patch_idx"])
                patch_anomaly_mask, patch_anomaly_ratio = point_mask_to_patch_targets(
                    labels, features["patch_idx"], args.mode_anomaly_patch_threshold
                )
                patch_routing_weights = (
                    geometry["node_mode_weights"]
                    if args.use_patch_mode_routing
                    else geometry["mode_weights"]
                )
                patch_gate_logits = (
                    geometry["abnormal_gate_logits"]
                    if args.use_geometry_abnormal_gate else None
                )
                gate_top_count = max(
                    1, int(geometry["abnormal_gate_logits"].shape[1] * args.top_percent)
                )
                sample_gate_logits = (
                    geometry["abnormal_gate_logits"].topk(
                        gate_top_count, dim=1
                    ).values.mean(dim=1)
                    if args.use_geometry_abnormal_gate else None
                )
                mode_prompts = encode_geometric_mode_prompts(
                    mode_prompt_model, encoder.open_clip_model, object_names, geometry["delta_A"]
                )
                patch_logits = mode_aware_anomaly_logits(
                    patch_embeddings,
                    mode_prompts["normal_text_embed"],
                    mode_prompts["dynamic_abnormal_text_embeds"],
                    patch_routing_weights,
                    args.temperature,
                    args.mode_score_type,
                    args.use_mode_weighted_scoring,
                    args.mode_score_temperature,
                    patch_gate_logits,
                    args.geometry_gate_logit_scale,
                )
                global_logits = mode_aware_anomaly_logits(
                    features["concat"].unsqueeze(1),
                    mode_prompts["normal_text_embed"],
                    mode_prompts["dynamic_abnormal_text_embeds"],
                    geometry["mode_weights"],
                    args.temperature,
                    args.mode_score_type,
                    args.use_mode_weighted_scoring,
                    args.mode_score_temperature,
                    sample_gate_logits,
                    args.geometry_gate_logit_scale,
                ).squeeze(1)
                if args.use_mode_diversity_loss:
                    diversity = mode_diversity_loss(
                        mode_prompts["dynamic_abnormal_text_embeds"]
                    )
                if args.use_mode_entropy_loss:
                    entropy_loss = mode_entropy_regularization(
                        geometry["node_mode_weights"],
                        args.mode_conditional_entropy_weight,
                        patch_anomaly_mask,
                    )
                if args.use_sinkhorn_mode_assignment:
                    assignment_loss = sinkhorn_mode_assignment_loss(
                        geometry["node_mode_logits"], patch_anomaly_mask,
                        args.sinkhorn_epsilon, args.sinkhorn_iterations,
                        args.mode_router_temperature,
                    )
                if args.use_geometry_abnormal_gate:
                    gate_targets = patch_anomaly_mask.to(
                        geometry["abnormal_gate_logits"].dtype
                    )
                    gate_loss = geometry_gate_supervision_loss(
                        geometry["abnormal_gate_logits"], gate_targets,
                        args.geometry_gate_margin,
                    )
                if args.use_residual_suppression_loss:
                    suppression = residual_suppression_loss(
                        geometry["delta_A"], object_labels
                    )
                if args.lambda_se3_sanity > 0:
                    transformed = mode_prompt_model.forward_geometry(
                        random_mode_se3_transform(points), features["patch_idx"]
                    )
                    sanity = se3_sanity_loss(
                        geometry["mode_weights"], transformed["mode_weights"],
                        geometry["delta_A"], transformed["delta_A"],
                    )
                with torch.no_grad():
                    weights = geometry["mode_weights"]
                    entropy_values = normalized_mode_entropy(weights)
                    delta_per_sample = geometry["delta_A"].norm(dim=-1).mean(dim=(1, 2))
                    mode_usage_sum += weights.sum(dim=0)
                    mode_usage_sq_sum += weights.square().sum(dim=0)
                    mode_sample_count += points.shape[0]
                    mode_max_sum += float(weights.max(dim=-1).values.sum())
                    mode_entropy_sum += float(entropy_values.sum())
                    delta_norm_sum += float(delta_per_sample.sum())
                    normal_mask = object_labels <= 0
                    abnormal_mask = object_labels > 0
                    if normal_mask.any():
                        normal_delta_sum += float(delta_per_sample[normal_mask].sum())
                        normal_delta_count += int(normal_mask.sum())
                    if abnormal_mask.any():
                        abnormal_delta_sum += float(delta_per_sample[abnormal_mask].sum())
                        abnormal_delta_count += int(abnormal_mask.sum())
                    normal_patch_mask = ~patch_anomaly_mask
                    if patch_anomaly_mask.any():
                        selected = geometry["node_mode_weights"][patch_anomaly_mask]
                        anomaly_patch_usage_sum += selected.sum(dim=0)
                        anomaly_patch_count += selected.shape[0]
                        anomaly_patch_entropy_sum += float(
                            normalized_mode_entropy(selected).sum()
                        )
                        anomaly_patch_confidence_sum += float(
                            selected.max(dim=-1).values.sum()
                        )
                        anomaly_gate_probability_sum += float(
                            geometry["abnormal_gate_logits"][patch_anomaly_mask]
                            .sigmoid().sum()
                        )
                    if normal_patch_mask.any():
                        selected = geometry["node_mode_weights"][normal_patch_mask]
                        normal_patch_usage_sum += selected.sum(dim=0)
                        normal_patch_count += selected.shape[0]
                        normal_patch_entropy_sum += float(
                            normalized_mode_entropy(selected).sum()
                        )
                        normal_patch_confidence_sum += float(
                            selected.max(dim=-1).values.sum()
                        )
                        normal_gate_probability_sum += float(
                            geometry["abnormal_gate_logits"][normal_patch_mask]
                            .sigmoid().sum()
                        )
            elif prompt_learner is None:
                patch_logits = patch_text_logits(
                    patch_embeddings, fixed_normal, fixed_anomaly, args.temperature
                )
                global_logits = patch_text_logits(
                    features["concat"].unsqueeze(1),
                    fixed_normal, fixed_anomaly, args.temperature,
                ).squeeze(1)
            else:
                base_prompts = encode_geometric_prompts(
                    prompt_learner, encoder.open_clip_model, object_names=object_names
                )
                dynamic_proto = None
                if prior_network is not None:
                    prior_result = prior_network(
                        patch_embeddings, base_prompts["normal_text_embed"],
                        base_prompts["abnormal_text_proto"], points, features["patch_idx"],
                    )
                    if args.lambda_geometry_invariance > 0:
                        transformed_points = random_se3_transform(points)
                        transformed_prior = prior_network(
                            patch_embeddings.detach(), base_prompts["normal_text_embed"],
                            base_prompts["abnormal_text_proto"], transformed_points,
                            features["patch_idx"],
                        )
                        invariance_loss = geometric_prior_invariance_loss(
                            prior_result["prior"], transformed_prior["prior"]
                        )
                    dynamic_prompts = encode_prior_enabled_abnormal_prompts(
                        prompt_learner, encoder.open_clip_model, prior_result["prior"],
                        object_names=object_names,
                    )
                    dynamic_proto = dynamic_prompts["prior_enabled_abnormal_text_proto"]
                    prior_loss = normal_sample_prior_loss(
                        prior_result["prior"], object_labels
                    )
                patch_logits = geometric_anomaly_logits(
                    patch_embeddings, base_prompts["normal_text_embed"],
                    base_prompts["abnormal_text_proto"], dynamic_proto, args.temperature,
                )
                global_logits = geometric_anomaly_logits(
                    features["concat"].unsqueeze(1), base_prompts["normal_text_embed"],
                    base_prompts["abnormal_text_proto"], dynamic_proto, args.temperature,
                ).squeeze(1)
                orthogonal = abnormal_prompt_orthogonal_loss(
                    base_prompts["abnormal_text_embeds"]
                )

            point_logits = patch_to_point(
                patch_logits, features["patch_idx"], labels.shape[1]
            )
            object_probability = aggregate_object_probability(
                global_logits, patch_logits, args.global_alpha, args.top_percent
            )
            local = (
                args.focal_weight * focal_loss(point_logits, labels)
                + args.dice_weight * dice_loss(point_logits, labels)
            )
            object_value = object_loss(
                object_probability, object_labels.to(object_probability.dtype)
            )
            total = local + args.object_weight * object_value
            if mode_prompt_model is not None:
                if args.use_mode_diversity_loss:
                    total = total + args.lambda_mode_diversity * diversity
                if args.use_mode_entropy_loss:
                    total = total + args.lambda_mode_entropy * entropy_loss
                if args.use_sinkhorn_mode_assignment:
                    total = total + args.lambda_mode_assignment * assignment_loss
                if args.use_geometry_abnormal_gate:
                    total = total + args.lambda_geometry_gate * gate_loss
                if args.use_residual_suppression_loss:
                    total = total + args.lambda_residual_suppression * suppression
                total = total + args.lambda_se3_sanity * sanity
            elif prompt_learner is not None:
                total = total + args.lambda_prompt_orthogonal * orthogonal
                total = total + args.lambda_prior * prior_loss
                total = total + args.lambda_geometry_invariance * invariance_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if args.gradient_clip_norm > 0:
                trainable_parameters = [
                    parameter for group in optimizer.param_groups
                    for parameter in group["params"] if parameter.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.gradient_clip_norm
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            for name, value in (
                ("local", local), ("object", object_value),
                ("orthogonal", orthogonal), ("prior", prior_loss),
                ("invariance", invariance_loss), ("mode_diversity", diversity),
                ("mode_entropy", entropy_loss), ("mode_assignment", assignment_loss),
                ("geometry_gate", gate_loss), ("residual_suppression", suppression),
                ("se3_sanity", sanity), ("total", total),
            ):
                totals[name] += float(value.detach())

        averages = {name: value / max(1, len(loader)) for name, value in totals.items()}
        if mode_prompt_model is not None:
            display_names = ["local", "object"]
            if args.use_mode_diversity_loss:
                display_names.append("mode_diversity")
            if args.use_mode_entropy_loss:
                display_names.append("mode_entropy")
            if args.use_sinkhorn_mode_assignment:
                display_names.append("mode_assignment")
            if args.use_geometry_abnormal_gate:
                display_names.append("geometry_gate")
            if args.use_residual_suppression_loss:
                display_names.append("residual_suppression")
            if args.lambda_se3_sanity > 0:
                display_names.append("se3_sanity")
        else:
            display_names = ["local", "object", "orthogonal", "prior"]
            if args.lambda_geometry_invariance > 0:
                display_names.append("invariance")
        display_names.append("total")
        epoch_message = f"Epoch {epoch + 1}/{epochs} | " + " | ".join(
            f"{name}={averages[name]:.6f}" for name in display_names
        )
        epoch_message += f" | lr={optimizer.param_groups[0]['lr']:.8f}"
        if mode_prompt_model is not None:
            count = max(1, mode_sample_count)
            usage_tensor = mode_usage_sum / count
            usage_std_tensor = (
                mode_usage_sq_sum / count - usage_tensor.square()
            ).clamp_min(0).sqrt()
            conditional_entropy = mode_entropy_sum / count
            marginal_entropy = float(
                -(usage_tensor * usage_tensor.clamp_min(1e-8).log()).sum()
                / np.log(args.num_geometric_modes)
            )
            mode_information = marginal_entropy - conditional_entropy
            usage = usage_tensor.detach().cpu().tolist()
            usage_std = usage_std_tensor.detach().cpu().tolist()
            anomaly_usage = (
                anomaly_patch_usage_sum / max(1, anomaly_patch_count)
            ).detach().cpu().tolist()
            normal_usage = (
                normal_patch_usage_sum / max(1, normal_patch_count)
            ).detach().cpu().tolist()
            epoch_message += (
                f" | mode_weights_mean={1.0 / args.num_geometric_modes:.6f}"
                f" | mode_weights_max={mode_max_sum / count:.6f}"
                f" | mode_conditional_entropy={conditional_entropy:.6f}"
                f" | mode_marginal_entropy={marginal_entropy:.6f}"
                f" | mode_information={mode_information:.6f}"
                f" | mode_usage={[round(value, 6) for value in usage]}"
                f" | mode_usage_std={[round(value, 6) for value in usage_std]}"
                f" | delta_A_norm={delta_norm_sum / count:.6f}"
                f" | normal_delta_A_norm={normal_delta_sum / max(1, normal_delta_count):.6f}"
                f" | abnormal_delta_A_norm={abnormal_delta_sum / max(1, abnormal_delta_count):.6f}"
                f" | anomaly_patch_entropy={anomaly_patch_entropy_sum / max(1, anomaly_patch_count):.6f}"
                f" | anomaly_patch_confidence={anomaly_patch_confidence_sum / max(1, anomaly_patch_count):.6f}"
                f" | normal_patch_entropy={normal_patch_entropy_sum / max(1, normal_patch_count):.6f}"
                f" | normal_patch_confidence={normal_patch_confidence_sum / max(1, normal_patch_count):.6f}"
                f" | anomaly_gate_probability={anomaly_gate_probability_sum / max(1, anomaly_patch_count):.6f}"
                f" | normal_gate_probability={normal_gate_probability_sum / max(1, normal_patch_count):.6f}"
                f" | anomaly_patch_usage={[round(value, 6) for value in anomaly_usage]}"
                f" | normal_patch_usage={[round(value, 6) for value in normal_usage]}"
            )
        log_line(train_log, epoch_message)
        if averages["total"] < best_loss:
            best_loss = averages["total"]
            state = {
                "adapter": adapter.state_dict(), "train_category": args.train_category,
                "feature_layers": list(args.feature_layers), "token_dim": args.token_dim,
                "text_dim": args.text_dim, "epoch": epoch + 1, "loss": best_loss,
                "use_geometric_cap": args.use_geometric_cap,
                "use_geometric_dap": args.use_geometric_dap,
                "use_geometric_mode_prompt": args.use_geometric_mode_prompt,
                "geometric_mode_version": args.geometric_mode_version if args.use_geometric_mode_prompt else None,
                "num_geometric_modes": args.num_geometric_modes if args.use_geometric_mode_prompt else None,
                "prompt_template": args.prompt_template if args.use_geometric_mode_prompt else None,
                "use_category_prompt": args.use_category_prompt if args.use_geometric_mode_prompt else None,
                "use_patch_mode_routing": args.use_patch_mode_routing if args.use_geometric_mode_prompt else None,
                "mode_anomaly_patch_threshold": args.mode_anomaly_patch_threshold if args.use_geometric_mode_prompt else None,
                "use_geometry_abnormal_gate": args.use_geometry_abnormal_gate if args.use_geometric_mode_prompt else None,
                "use_sinkhorn_mode_assignment": args.use_sinkhorn_mode_assignment if args.use_geometric_mode_prompt else None,
                "mode_score_type": args.mode_score_type if args.use_geometric_mode_prompt else None,
                "dap_version": "se3_graph_v1" if args.use_geometric_dap else None,
                "prompt_suffix_mode": "prompt_template_v1" if args.use_geometric_cap else None,
                "geometric_prompt_suffix": args.geometric_prompt_suffix if args.use_geometric_cap else None,
            }
            if prompt_learner is not None:
                state["geometric_cap"] = prompt_learner.state_dict()
            if prior_network is not None:
                state["geometric_dap"] = prior_network.state_dict()
            if mode_prompt_model is not None:
                state["geometric_mode_prompt"] = mode_prompt_model.state_dict()
            torch.save(state, run_dir / "best.pth")
    log_line(train_log, f"Best checkpoint: {run_dir / 'best.pth'} | loss={best_loss:.6f}")
    write_yaml(
        run_dir / "training_complete.yaml",
        {
            "train_category": args.train_category,
            "dataset_name": args.dataset_name,
            "epochs": epochs,
            "best_loss": float(best_loss),
            "use_geometric_mode_prompt": args.use_geometric_mode_prompt,
            "geometric_mode_version": args.geometric_mode_version if args.use_geometric_mode_prompt else None,
            "prompt_template": args.prompt_template if args.use_geometric_mode_prompt else None,
            "use_patch_mode_routing": args.use_patch_mode_routing if args.use_geometric_mode_prompt else None,
            "mode_anomaly_patch_threshold": args.mode_anomaly_patch_threshold if args.use_geometric_mode_prompt else None,
            "use_geometry_abnormal_gate": args.use_geometry_abnormal_gate if args.use_geometric_mode_prompt else None,
            "use_sinkhorn_mode_assignment": args.use_sinkhorn_mode_assignment if args.use_geometric_mode_prompt else None,
            "mode_score_type": args.mode_score_type if args.use_geometric_mode_prompt else None,
            "dap_version": "se3_graph_v1" if args.use_geometric_dap else None,
            "prompt_suffix_mode": "prompt_template_v1" if args.use_geometric_cap else None,
            "geometric_prompt_suffix": args.geometric_prompt_suffix if args.use_geometric_cap else None,
            "complete": True,
        },
    )


if __name__ == "__main__":
    main()
