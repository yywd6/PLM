"""Trainable point-language adapters without geometric visual fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearPatchAdapter(nn.Module):
    """Single-layer adapter retained for compatibility and ablations."""

    def __init__(self, input_dim=384, output_dim=1280):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.normalization = nn.LayerNorm(output_dim)

    def forward(self, patch_tokens):
        return F.normalize(self.normalization(self.projection(patch_tokens)), dim=-1)


class MultiLayerPatchAdapter(nn.Module):
    """Project ULIP layers independently and fuse them with learned scalar weights."""

    def __init__(self, input_dim=384, output_dim=1280, num_layers=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.adapters = nn.ModuleList(
            [LinearPatchAdapter(input_dim, output_dim) for _ in range(num_layers)]
        )
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self, layer_tokens):
        if len(layer_tokens) != len(self.adapters):
            raise ValueError(
                f"Expected {len(self.adapters)} feature layers, got {len(layer_tokens)}"
            )
        projected = torch.stack(
            [adapter(tokens) for adapter, tokens in zip(self.adapters, layer_tokens)],
            dim=0,
        )
        weights = torch.softmax(self.layer_logits, dim=0)
        fused = (projected * weights[:, None, None, None]).sum(dim=0)
        return F.normalize(fused, dim=-1)

    def layer_weights(self):
        return torch.softmax(self.layer_logits, dim=0)


def select_patch_tokens(layer_features, patch_indices, feature_layer):
    tokens = layer_features[feature_layer]
    patch_count = patch_indices.shape[1]
    if tokens.shape[1] == patch_count + 1:
        tokens = tokens[:, 1:, :]
    if tokens.shape[1] != patch_count:
        raise ValueError(
            f"Layer {feature_layer} has {tokens.shape[1]} tokens, expected {patch_count} patches"
        )
    return tokens.contiguous()


def select_multi_layer_tokens(layer_features, patch_indices, feature_layers):
    return [
        select_patch_tokens(layer_features, patch_indices, layer)
        for layer in feature_layers
    ]


def patch_text_logits(patch_embeddings, normal_embedding, anomaly_embedding, temperature=0.07):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    patch_embeddings = F.normalize(patch_embeddings, dim=-1)
    normal_embedding = F.normalize(normal_embedding, dim=-1)
    anomaly_embedding = F.normalize(anomaly_embedding, dim=-1)
    normal = torch.matmul(patch_embeddings, normal_embedding.T).squeeze(-1)
    anomaly = torch.matmul(patch_embeddings, anomaly_embedding.T).squeeze(-1)
    return (anomaly - normal) / temperature


def patch_to_point(patch_values, patch_indices, num_points):
    batch_size, _, neighborhood_size = patch_indices.shape
    sums = torch.zeros(batch_size, num_points, device=patch_values.device)
    counts = torch.zeros_like(sums)
    for batch_index in range(batch_size):
        values = patch_values[batch_index].unsqueeze(-1).expand(-1, neighborhood_size)
        indices = patch_indices[batch_index].long()
        sums[batch_index].index_add_(0, indices.reshape(-1), values.reshape(-1))
        counts[batch_index].index_add_(0, indices.reshape(-1), torch.ones_like(values).reshape(-1))
    return sums / counts.clamp_min(1.0)


def pool_patch_logits(patch_logits, top_percent=0.2):
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    count = max(1, int(patch_logits.shape[1] * top_percent))
    return torch.topk(patch_logits, k=count, dim=1).values.mean(dim=1)


def aggregate_object_probability(global_logits, patch_logits, global_alpha=0.5, top_percent=0.2):
    """Combine ULIP global and top-k local anomaly probabilities."""
    if not 0.0 <= global_alpha <= 1.0:
        raise ValueError("global_alpha must be in [0, 1]")
    if not 0 < top_percent <= 1:
        raise ValueError("top_percent must be in (0, 1]")
    if global_logits.ndim != 1:
        raise ValueError("global_logits must have shape [B]")
    if patch_logits.ndim != 2 or patch_logits.shape[0] != global_logits.shape[0]:
        raise ValueError("patch_logits must have shape [B, G]")

    # Float64 avoids sigmoid saturation for similarity logits near +/-30.
    global_prob = torch.sigmoid(global_logits.double())
    patch_prob = torch.sigmoid(patch_logits.double())
    count = max(1, int(patch_prob.shape[1] * top_percent))
    local_prob = torch.topk(patch_prob, k=count, dim=1).values.mean(dim=1)
    return global_alpha * global_prob + (1.0 - global_alpha) * local_prob
