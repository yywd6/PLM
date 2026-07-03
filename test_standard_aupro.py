"""Evaluate baseline or 3D geometric CAP/DAP on one-rest targets."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.anomaly_datasets import PointCloudDataset
from evaluation_metrics import finite_mean, point_aupro, safe_binary_metrics
from models.geometric_cap import (
    GeometricCompoundPromptLearner,
    encode_geometric_prompts,
    geometric_anomaly_logits,
)
from models.geometric_dap import PointCloudAbnormalityPrior, compute_patch_geometry_descriptor
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
    write_json,
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
    parser = argparse.ArgumentParser(description="Evaluate ULIP baseline or 3D-CAP/DAP.")
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_test_samples_per_category", type=int, default=0)
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
    recognized = {action.dest for action in parser._actions}
    parser.set_defaults(**{key: value for key, value in config.items() if key in recognized})
    args = parser.parse_args()
    if not args.train_category:
        parser.error("--train_category is required")
    validate_one_rest_flags(args)
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


def build_prompt_modules(args, encoder, checkpoint, device):
    if not checkpoint.get("use_geometric_cap", False):
        return None, None
    prompt = GeometricCompoundPromptLearner(
        encoder.open_clip_model,
        encoder.tokenizer,
        args.num_geometric_abnormal_prompts,
        args.num_normal_tokens,
        args.num_abnormal_tokens,
        args.geometric_prompt_suffix,
    ).to(device)
    prompt.load_state_dict(checkpoint["geometric_cap"])
    prompt.eval()
    prior = None
    if checkpoint.get("use_geometric_dap", False):
        prior = PointCloudAbnormalityPrior(
            feature_dim=args.text_dim,
            text_dim=args.text_dim,
            prompt_token_dim=prompt.token_width,
            geo_dim=args.patch_geo_desc_dim,
            hidden_dim=args.prior_hidden_dim,
            top_m=args.top_m_abnormal_patches,
        ).to(device)
        prior.load_state_dict(checkpoint["geometric_dap"])
        prior.eval()
    return prompt, prior


@torch.inference_mode()
def main():
    args = parse_args()
    train_categories, test_categories = resolve_categories(
        args.dataset_name, args.protocol, args.train_category
    )
    run_dir = Path(args.output_root) / args.train_category
    run_dir.mkdir(parents=True, exist_ok=True)
    test_log = run_dir / "test.log"
    test_log.write_text("", encoding="utf-8")
    log_line(test_log, f"Protocol: {args.protocol}")
    log_line(test_log, f"Train categories: {train_categories}")
    log_line(test_log, f"Test categories: {test_categories}")
    dataset = PointCloudDataset(
        args.data_root, split=args.test_split, classes=test_categories,
        dataset_name=args.dataset_name,
    )
    if len(dataset) == 0:
        raise FileNotFoundError("No target test samples")
    observed = assert_dataset_categories(dataset, test_categories, train_categories)
    log_line(test_log, f"Observed test path categories: {observed}")
    limit = args.max_test_samples_per_category
    if args.debug:
        limit = limit or 4
        log_line(test_log, f"Debug mode: max_test_samples_per_category={limit}")
    loader = DataLoader(
        subset_per_category(dataset, limit), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("train_category") != args.train_category:
        raise RuntimeError("Checkpoint train_category mismatch")
    if "feature_layers" not in checkpoint:
        raise RuntimeError(
            "Checkpoint uses the old single-layer adapter; retrain with feature_layers=[2,5,8,11]"
        )
    use_cap = checkpoint.get("use_geometric_cap", False)
    use_dap = checkpoint.get("use_geometric_dap", False)
    log_line(test_log, f"Geometric CAP={use_cap}, DAP={use_dap}")
    encoder = ULIP2Encoder(
        args.model_path, device=device, num_points=args.num_points,
        return_layers=tuple(args.return_layers), return_clip=use_cap,
    )
    feature_layers = checkpoint.get("feature_layers", args.feature_layers)
    adapter = MultiLayerPatchAdapter(
        checkpoint.get("token_dim", args.token_dim),
        checkpoint.get("text_dim", args.text_dim),
        len(feature_layers),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter"])
    adapter.eval()
    prompt_learner, prior_network = build_prompt_modules(args, encoder, checkpoint, device)
    if prompt_learner is None:
        fixed_normal = encoder.encode_text_templates(args.normal_templates)
        fixed_anomaly = encoder.encode_text_templates(args.anomaly_templates)
    else:
        fixed_normal = fixed_anomaly = None

    values = defaultdict(lambda: defaultdict(list))
    category_names = PointCloudDataset.PRESETS[args.dataset_name]
    feature_layers = checkpoint.get("feature_layers", args.feature_layers)
    for batch in tqdm(loader, desc=f"Test source={args.train_category}", dynamic_ncols=True):
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"]
        features = encoder.encode_pointcloud(points, return_intermediate=True)
        tokens = select_multi_layer_tokens(
            features["layer_feats"], features["patch_idx"], feature_layers
        )
        patch_embeddings = adapter(tokens)
        if prompt_learner is None:
            patch_logits = patch_text_logits(
                patch_embeddings, fixed_normal, fixed_anomaly, args.temperature
            )
        else:
            base_prompts = encode_geometric_prompts(prompt_learner, encoder.open_clip_model)
            dynamic_proto = None
            if prior_network is not None:
                geometry = compute_patch_geometry_descriptor(points, features["patch_idx"])
                prior_result = prior_network(
                    patch_embeddings, base_prompts["normal_text_embed"],
                    base_prompts["abnormal_text_proto"], geometry,
                )
                dynamic = encode_geometric_prompts(
                    prompt_learner, encoder.open_clip_model, prior_result["prior"]
                )
                dynamic_proto = dynamic["prior_enabled_abnormal_text_proto"]
            patch_logits = geometric_anomaly_logits(
                patch_embeddings, base_prompts["normal_text_embed"],
                base_prompts["abnormal_text_proto"], dynamic_proto, args.temperature,
            )
        point_scores = torch.sigmoid(
            patch_to_point(patch_logits, features["patch_idx"], labels.shape[1])
        ).cpu().numpy()
        object_scores = torch.sigmoid(pool_patch_logits(patch_logits, args.top_percent)).cpu().numpy()
        for index, category in enumerate(batch["category"]):
            name = category_names[int(category)]
            point_labels = labels[index].numpy().reshape(-1)
            values[name]["object_labels"].append(int((point_labels > 0).any()))
            values[name]["object_scores"].append(float(object_scores[index]))
            values[name]["point_labels"].append(point_labels)
            values[name]["point_scores"].append(point_scores[index].reshape(-1))

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
    metric_names = ("object_auroc", "object_ap", "point_auroc", "point_ap", "point_pro")
    means = {key: finite_mean([item[key] for item in per_category.values()]) for key in metric_names}
    if args.save_per_category_metrics:
        write_json(run_dir / "per_category_metrics.json", {
            "train_category": args.train_category, "test_categories": test_categories,
            "use_geometric_cap": use_cap, "use_geometric_dap": use_dap,
            "metrics": per_category,
        })
    if args.save_mean_metrics:
        write_json(run_dir / "mean_metrics.json", {
            "train_category": args.train_category, "test_categories": test_categories, **means,
        })
    log_line(test_log, f"Mean metrics: {means}")


if __name__ == "__main__":
    main()
