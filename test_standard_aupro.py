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
    encode_prior_enabled_abnormal_prompts,
    geometric_anomaly_logits,
)
from models.geometric_dap import PointCloudAbnormalityPrior
from models.geometric_mode_prompt import (
    GeometricModePromptLearner,
    encode_geometric_mode_prompts,
    format_category_prompt,
    mode_aware_anomaly_logits,
    normalized_mode_entropy,
    point_mask_to_patch_targets,
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
    parser.add_argument("--global_alpha", type=float, default=0.5)
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
            graph_dim=args.geometry_graph_dim,
            hidden_dim=args.prior_hidden_dim,
            graph_k=args.geometry_graph_k,
            graph_layers=args.geometry_graph_layers,
            routing_temperature=args.geometry_routing_temperature,
            top_m=args.top_m_abnormal_patches,
        ).to(device)
        prior.load_state_dict(checkpoint["geometric_dap"])
        prior.eval()
    return prompt, prior


def build_geometric_mode_model(args, encoder, checkpoint, device):
    if not checkpoint.get("use_geometric_mode_prompt", False):
        return None
    if checkpoint.get("geometric_mode_version") != args.geometric_mode_version:
        raise RuntimeError("Checkpoint geometric mode prompting version is incompatible")
    if checkpoint.get("prompt_template") != args.prompt_template:
        raise RuntimeError("Checkpoint prompt_template does not match config")
    model = GeometricModePromptLearner(
        clip_model=encoder.open_clip_model,
        tokenizer=encoder.tokenizer,
        num_modes=checkpoint.get("num_geometric_modes", args.num_geometric_modes),
        num_normal_tokens=args.num_normal_tokens,
        num_abnormal_tokens=args.num_abnormal_tokens,
        prompt_template=args.prompt_template,
        use_category_prompt=checkpoint.get("use_category_prompt", args.use_category_prompt),
        graph_dim=args.geometry_graph_dim,
        graph_k=args.geometry_graph_k,
        graph_layers=args.geometry_graph_layers,
        router_temperature=args.mode_router_temperature,
        residual_scale=args.mode_residual_scale,
        modulator_hidden_dim=args.mode_prompt_hidden_dim,
        use_mode_specific_residual=args.use_mode_specific_residual,
    ).to(device)
    load_result = model.load_state_dict(
        checkpoint["geometric_mode_prompt"], strict=False
    )
    if args.geometric_mode_version != "se3_mode_prompt_v6_patch_routing":
        if load_result.missing_keys or load_result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint state mismatch: missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )
    else:
        invalid_missing = [
            key for key in load_result.missing_keys
            if not key.startswith("mode_router.abnormal_gate.")
        ]
        if invalid_missing or load_result.unexpected_keys:
            raise RuntimeError(
                f"Legacy v6 checkpoint state mismatch: missing={invalid_missing}, "
                f"unexpected={load_result.unexpected_keys}"
            )
    model.eval()
    return model


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
    use_mode_prompt = checkpoint.get("use_geometric_mode_prompt", False)
    if use_mode_prompt and (use_cap or use_dap):
        raise RuntimeError("Checkpoint mixes legacy CAP/DAP with geometric mode prompting")
    if use_cap and checkpoint.get("prompt_suffix_mode") != "prompt_template_v1":
        raise RuntimeError(
            "Checkpoint prompt suffix mode is incompatible; retrain with the current prompt template implementation"
        )
    if use_cap and checkpoint.get("geometric_prompt_suffix") != args.geometric_prompt_suffix:
        raise RuntimeError("Checkpoint geometric prompt template does not match config")
    if use_dap and checkpoint.get("dap_version") != "se3_graph_v1":
        raise RuntimeError(
            "Checkpoint uses the old 4D/top-M DAP; retrain with the SE(3) graph configuration"
        )
    if use_mode_prompt:
        log_line(test_log, "GeometricModePrompt=True")
        log_line(test_log, f"Prompt template: {args.prompt_template}")
    else:
        log_line(test_log, f"Geometric CAP={use_cap}, DAP={use_dap}")
    encoder = ULIP2Encoder(
        args.model_path, device=device, num_points=args.num_points,
        return_layers=tuple(args.return_layers), return_clip=use_cap or use_mode_prompt,
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
    mode_prompt_model = build_geometric_mode_model(args, encoder, checkpoint, device)
    if prompt_learner is None and mode_prompt_model is None:
        fixed_normal = encoder.encode_text_templates(args.normal_templates)
        fixed_anomaly = encoder.encode_text_templates(args.anomaly_templates)
    else:
        fixed_normal = fixed_anomaly = None

    values = defaultdict(lambda: defaultdict(list))
    mode_values = defaultdict(lambda: defaultdict(list))
    category_names = PointCloudDataset.PRESETS[args.dataset_name]
    feature_layers = checkpoint.get("feature_layers", args.feature_layers)
    for batch in tqdm(loader, desc=f"Test source={args.train_category}", dynamic_ncols=True):
        points = batch["points"].to(device, non_blocking=True)
        labels = batch["labels"]
        object_names = [category_names[int(category)] for category in batch["category"]]
        features = encoder.encode_pointcloud(points, return_intermediate=True)
        tokens = select_multi_layer_tokens(
            features["layer_feats"], features["patch_idx"], feature_layers
        )
        patch_embeddings = adapter(tokens)
        batch_mode_weights = batch_mode_entropy = batch_delta_norm = None
        batch_node_weights = batch_patch_targets = batch_gate_probabilities = None
        if mode_prompt_model is not None:
            geometry = mode_prompt_model.forward_geometry(points, features["patch_idx"])
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
                patch_embeddings, mode_prompts["normal_text_embed"],
                mode_prompts["dynamic_abnormal_text_embeds"], (geometry["node_mode_weights"] if args.use_patch_mode_routing else geometry["mode_weights"]),
                args.temperature, args.mode_score_type, args.use_mode_weighted_scoring,
                args.mode_score_temperature, patch_gate_logits,
                args.geometry_gate_logit_scale,
            )
            global_logits = mode_aware_anomaly_logits(
                features["concat"].unsqueeze(1), mode_prompts["normal_text_embed"],
                mode_prompts["dynamic_abnormal_text_embeds"], geometry["mode_weights"],
                args.temperature, args.mode_score_type, args.use_mode_weighted_scoring,
                args.mode_score_temperature, sample_gate_logits,
                args.geometry_gate_logit_scale,
            ).squeeze(1)
            patch_targets, _ = point_mask_to_patch_targets(
                labels.to(device), features["patch_idx"], args.mode_anomaly_patch_threshold
            )
            batch_node_weights = geometry["node_mode_weights"].cpu().numpy()
            batch_patch_targets = patch_targets.cpu().numpy()
            batch_gate_probabilities = (
                geometry["abnormal_gate_logits"].sigmoid().cpu().numpy()
                if args.use_geometry_abnormal_gate else None
            )
            batch_mode_weights = geometry["mode_weights"].cpu().numpy()
            batch_mode_entropy = normalized_mode_entropy(geometry["mode_weights"]).cpu().numpy()
            batch_delta_norm = geometry["delta_A"].norm(dim=-1).mean(dim=(1, 2)).cpu().numpy()
        elif prompt_learner is None:
            patch_logits = patch_text_logits(
                patch_embeddings, fixed_normal, fixed_anomaly, args.temperature
            )
            global_logits = patch_text_logits(
                features["concat"].unsqueeze(1),
                fixed_normal,
                fixed_anomaly,
                args.temperature,
            ).squeeze(1)
        else:
            base_prompts = encode_geometric_prompts(
                prompt_learner, encoder.open_clip_model, object_names=object_names
            )
            dynamic_proto = None
            if prior_network is not None:
                prior_result = prior_network(
                    patch_embeddings, base_prompts["normal_text_embed"],
                    base_prompts["abnormal_text_proto"], points,
                    features["patch_idx"],
                )
                dynamic = encode_prior_enabled_abnormal_prompts(
                    prompt_learner, encoder.open_clip_model, prior_result["prior"],
                    object_names=object_names,
                )
                dynamic_proto = dynamic["prior_enabled_abnormal_text_proto"]
            patch_logits = geometric_anomaly_logits(
                patch_embeddings, base_prompts["normal_text_embed"],
                base_prompts["abnormal_text_proto"], dynamic_proto, args.temperature,
            )
            global_logits = geometric_anomaly_logits(
                features["concat"].unsqueeze(1),
                base_prompts["normal_text_embed"],
                base_prompts["abnormal_text_proto"], dynamic_proto, args.temperature,
            ).squeeze(1)
        # Keep raw logits for ranking metrics. Float32 sigmoid saturates large
        # top-k values to exactly 1.0 and creates artificial score ties.
        point_scores = patch_to_point(
            patch_logits, features["patch_idx"], labels.shape[1]
        ).cpu().numpy()
        object_scores = aggregate_object_probability(
            global_logits, patch_logits, args.global_alpha, args.top_percent
        ).cpu().numpy()
        for index, category in enumerate(batch["category"]):
            name = category_names[int(category)]
            point_labels = labels[index].numpy().reshape(-1)
            values[name]["object_labels"].append(int((point_labels > 0).any()))
            values[name]["object_scores"].append(float(object_scores[index]))
            values[name]["point_labels"].append(point_labels)
            values[name]["point_scores"].append(point_scores[index].reshape(-1))
            if batch_mode_weights is not None:
                mode_values[name]["weights"].append(batch_mode_weights[index])
                mode_values[name]["entropy"].append(float(batch_mode_entropy[index]))
                mode_values[name]["delta_norm"].append(float(batch_delta_norm[index]))
                mode_values[name]["node_weights"].append(batch_node_weights[index])
                mode_values[name]["patch_targets"].append(batch_patch_targets[index])
                if batch_gate_probabilities is not None:
                    mode_values[name]["gate_probabilities"].append(
                        batch_gate_probabilities[index]
                    )

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

    mode_statistics = None
    if mode_prompt_model is not None:
        category_statistics = []
        for name in test_categories:
            weights = np.asarray(mode_values[name]["weights"], dtype=np.float64)
            usage = weights.mean(axis=0) if weights.size else np.zeros(args.num_geometric_modes)
            usage_std = weights.std(axis=0) if weights.size else np.zeros(args.num_geometric_modes)
            node_weights = np.asarray(
                mode_values[name]["node_weights"], dtype=np.float64
            ).reshape(-1, args.num_geometric_modes)
            patch_targets = np.asarray(
                mode_values[name]["patch_targets"], dtype=bool
            ).reshape(-1)
            gate_probabilities = (
                np.asarray(
                    mode_values[name]["gate_probabilities"], dtype=np.float64
                ).reshape(-1)
                if args.use_geometry_abnormal_gate else np.array([], dtype=np.float64)
            )
            node_usage = (
                node_weights.mean(axis=0)
                if node_weights.size else np.zeros(args.num_geometric_modes)
            )
            node_entropy = float(
                -(node_weights * np.log(np.clip(node_weights, 1e-8, None))).sum(axis=-1).mean()
                / np.log(args.num_geometric_modes)
            ) if node_weights.size else None
            node_confidence = float(node_weights.max(axis=-1).mean()) if node_weights.size else None
            anomaly_node_usage = (
                node_weights[patch_targets].mean(axis=0)
                if patch_targets.any() else np.zeros(args.num_geometric_modes)
            )
            normal_node_usage = (
                node_weights[~patch_targets].mean(axis=0)
                if (~patch_targets).any() else np.zeros(args.num_geometric_modes)
            )
            anomaly_normal_route_gap = float(
                np.abs(anomaly_node_usage - normal_node_usage).mean()
            ) if patch_targets.any() and (~patch_targets).any() else None
            anomaly_gate_probability = (
                float(gate_probabilities[patch_targets].mean())
                if gate_probabilities.size and patch_targets.any() else None
            )
            normal_gate_probability = (
                float(gate_probabilities[~patch_targets].mean())
                if gate_probabilities.size and (~patch_targets).any() else None
            )
            gate_probability_gap = (
                anomaly_gate_probability - normal_gate_probability
                if anomaly_gate_probability is not None
                and normal_gate_probability is not None else None
            )
            entropy = finite_mean(mode_values[name]["entropy"])
            marginal_entropy = float(
                -(usage * np.log(np.clip(usage, 1e-8, None))).sum()
                / np.log(args.num_geometric_modes)
            )
            mode_information = marginal_entropy - entropy
            delta_norm = finite_mean(mode_values[name]["delta_norm"])
            prompt_text = mode_prompt_model.prompt_texts([name])[0]
            item = {
                "target_category": name,
                "prompt_text": prompt_text,
                "mode_usage_distribution": usage.tolist(),
                "mode_usage_std": usage_std.tolist(),
                "mode_entropy": entropy,
                "mode_marginal_entropy": marginal_entropy,
                "mode_information": mode_information,
                "delta_A_norm_mean": delta_norm,
                "patch_mode_usage_distribution": node_usage.tolist(),
                "patch_mode_entropy": node_entropy,
                "patch_mode_confidence": node_confidence,
                "anomaly_patch_mode_usage": anomaly_node_usage.tolist(),
                "normal_patch_mode_usage": normal_node_usage.tolist(),
                "anomaly_normal_route_gap": anomaly_normal_route_gap,
                "anomaly_gate_probability": anomaly_gate_probability,
                "normal_gate_probability": normal_gate_probability,
                "gate_probability_gap": gate_probability_gap,
            }
            category_statistics.append(item)
            log_line(
                test_log,
                f"Mode statistics {name}: prompt={prompt_text!r}, "
                f"usage={[round(value, 6) for value in usage.tolist()]}, "
                f"usage_std={[round(value, 6) for value in usage_std.tolist()]}, "
                f"conditional_entropy={entropy:.6f}, marginal_entropy={marginal_entropy:.6f}, "
                f"mode_information={mode_information:.6f}, delta_A_norm={delta_norm:.6f}, "
                f"patch_entropy={node_entropy:.6f}, patch_confidence={node_confidence:.6f}, "
                f"anomaly_normal_route_gap={anomaly_normal_route_gap}, "
                f"gate_probability_gap={gate_probability_gap}",
            )
        mode_statistics = {
            "train_category": args.train_category,
            "num_geometric_modes": args.num_geometric_modes,
            "categories": category_statistics,
        }
        write_json(run_dir / "mode_statistics.json", mode_statistics)

    metric_names = ("object_auroc", "object_ap", "point_auroc", "point_ap", "point_pro")
    means = {key: finite_mean([item[key] for item in per_category.values()]) for key in metric_names}
    if args.save_per_category_metrics:
        write_json(run_dir / "per_category_metrics.json", {
            "train_category": args.train_category, "test_categories": test_categories,
            "use_geometric_cap": use_cap, "use_geometric_dap": use_dap,
            "use_geometric_mode_prompt": use_mode_prompt,
            "metrics": per_category,
        })
    if args.save_mean_metrics:
        write_json(run_dir / "mean_metrics.json", {
            "train_category": args.train_category, "test_categories": test_categories, **means,
        })
    log_line(test_log, f"Mean metrics: {means}")


if __name__ == "__main__":
    main()
