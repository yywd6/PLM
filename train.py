"""Train the linear baseline or prompt-only 3D geometric CAP/DAP."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from data.anomaly_datasets import PointCloudDataset
from loss import BinaryDiceLoss, BinaryFocalLoss
from models.geometric_cap import (
    GeometricCompoundPromptLearner,
    abnormal_prompt_orthogonal_loss,
    encode_geometric_prompts,
    geometric_anomaly_logits,
)
from models.geometric_dap import (
    PointCloudAbnormalityPrior,
    compute_patch_geometry_descriptor,
    normal_sample_prior_loss,
)
from models.trainable_baseline import (
    MultiLayerPatchAdapter,
    patch_text_logits,
    patch_to_point,
    pool_patch_logits,
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--prompt_learning_rate", type=float, default=1e-3)
    parser.add_argument("--prior_learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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
    parser.add_argument("--geometric_prompt_suffix", default="point cloud patch")
    parser.add_argument("--top_m_abnormal_patches", type=int, default=10)
    parser.add_argument("--prior_hidden_dim", type=int, default=512)
    parser.add_argument("--patch_geo_desc_dim", type=int, default=4)
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
    if args.use_geometric_cap and not (args.freeze_plm and args.freeze_text_encoder):
        parser.error("3D-CAP requires freeze_plm=true and freeze_text_encoder=true")
    validate_one_rest_flags(args)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            geo_dim=args.patch_geo_desc_dim,
            hidden_dim=args.prior_hidden_dim,
            top_m=args.top_m_abnormal_patches,
        ).to(device)
    return prompt_learner, prior_network


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
        return_layers=tuple(args.return_layers), return_clip=args.use_geometric_cap,
    )
    adapter = MultiLayerPatchAdapter(
        args.token_dim, args.text_dim, len(args.feature_layers)
    ).to(device)
    prompt_learner, prior_network = build_cap_modules(args, encoder, adapter, device)
    if args.use_geometric_cap:
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

    focal_loss = BinaryFocalLoss(args.focal_gamma, args.focal_alpha)
    dice_loss = BinaryDiceLoss()
    object_loss = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    for epoch in range(epochs):
        adapter.train(
            not args.use_geometric_cap or not args.freeze_visual_adapter
        )
        if prompt_learner is not None:
            prompt_learner.train()
        if prior_network is not None:
            prior_network.train()
        totals = {name: 0.0 for name in ("local", "object", "orthogonal", "prior", "total")}
        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            features = encoder.encode_pointcloud(points, return_intermediate=True)
            tokens = select_multi_layer_tokens(
                features["layer_feats"], features["patch_idx"], args.feature_layers
            )
            patch_embeddings = adapter(tokens)
            orthogonal = patch_embeddings.sum() * 0.0
            prior_loss = patch_embeddings.sum() * 0.0
            if prompt_learner is None:
                patch_logits = patch_text_logits(
                    patch_embeddings, fixed_normal, fixed_anomaly, args.temperature
                )
            else:
                base_prompts = encode_geometric_prompts(
                    prompt_learner, encoder.open_clip_model
                )
                dynamic_proto = None
                if prior_network is not None:
                    geometry = compute_patch_geometry_descriptor(points, features["patch_idx"])
                    prior_result = prior_network(
                        patch_embeddings,
                        base_prompts["normal_text_embed"],
                        base_prompts["abnormal_text_proto"],
                        geometry,
                    )
                    dynamic_prompts = encode_geometric_prompts(
                        prompt_learner, encoder.open_clip_model, prior_result["prior"]
                    )
                    dynamic_proto = dynamic_prompts["prior_enabled_abnormal_text_proto"]
                    object_labels = (labels > 0).any(dim=1).float()
                    prior_loss = normal_sample_prior_loss(prior_result["prior"], object_labels)
                patch_logits = geometric_anomaly_logits(
                    patch_embeddings,
                    base_prompts["normal_text_embed"],
                    base_prompts["abnormal_text_proto"],
                    dynamic_proto,
                    args.temperature,
                )
                orthogonal = abnormal_prompt_orthogonal_loss(
                    base_prompts["abnormal_text_embeds"]
                )
            point_logits = patch_to_point(patch_logits, features["patch_idx"], labels.shape[1])
            object_logits = pool_patch_logits(patch_logits, args.top_percent)
            object_labels = (labels > 0).any(dim=1).float()
            local = args.focal_weight * focal_loss(point_logits, labels) + args.dice_weight * dice_loss(point_logits, labels)
            object_value = object_loss(object_logits, object_labels)
            total = local + args.object_weight * object_value
            if prompt_learner is not None:
                total = total + args.lambda_prompt_orthogonal * orthogonal
                total = total + args.lambda_prior * prior_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            for name, value in (
                ("local", local), ("object", object_value), ("orthogonal", orthogonal),
                ("prior", prior_loss), ("total", total),
            ):
                totals[name] += float(value.detach())
        averages = {name: value / max(1, len(loader)) for name, value in totals.items()}
        log_line(train_log, f"Epoch {epoch + 1}/{epochs} | " + " | ".join(
            f"{name}={value:.6f}" for name, value in averages.items()
        ))
        if averages["total"] < best_loss:
            best_loss = averages["total"]
            state = {
                "adapter": adapter.state_dict(), "train_category": args.train_category,
                "feature_layers": list(args.feature_layers), "token_dim": args.token_dim,
                "text_dim": args.text_dim, "epoch": epoch + 1, "loss": best_loss,
                "use_geometric_cap": args.use_geometric_cap,
                "use_geometric_dap": args.use_geometric_dap,
            }
            if prompt_learner is not None:
                state["geometric_cap"] = prompt_learner.state_dict()
            if prior_network is not None:
                state["geometric_dap"] = prior_network.state_dict()
            torch.save(state, run_dir / "best.pth")
    log_line(train_log, f"Best checkpoint: {run_dir / 'best.pth'} | loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
