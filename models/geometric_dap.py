"""SE(3)-invariant patch-graph prior for dynamic abnormal prompts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


NODE_FEATURE_DIM = 7
EDGE_FEATURE_DIM = 5


def compute_invariant_patch_graph_inputs(points, patch_indices, graph_k=8, eps=1e-6):
    """Build invariant scalar node/edge features and a patch kNN graph."""
    xyz = points[..., :3]
    indices = patch_indices.long()
    batch_size, patch_count, neighborhood_size = indices.shape
    batch_indices = torch.arange(batch_size, device=points.device)[:, None, None]
    patches = xyz[batch_indices, indices]
    centers = patches.mean(dim=2)
    centered = patches - centers.unsqueeze(2)
    covariance = centered.transpose(-1, -2) @ centered
    covariance = covariance / max(1, neighborhood_size)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    eigenvalue_ratios = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(eps)
    curvature = eigenvalue_ratios[..., :1]
    normals = eigenvectors[..., :, 0]

    object_center = xyz.mean(dim=1, keepdim=True)
    object_scale = (
        (xyz - object_center).square().sum(dim=-1).mean(dim=1, keepdim=True).sqrt()
    ).clamp_min(eps)
    local_radius = centered.square().sum(dim=-1).mean(dim=-1).sqrt()
    normalized_radius = local_radius / object_scale
    log_density = -3.0 * torch.log(normalized_radius.clamp_min(eps))
    relative_density = log_density - log_density.mean(dim=1, keepdim=True)
    radial_distance = (centers - object_center).norm(dim=-1) / object_scale
    node_features = torch.cat(
        (
            eigenvalue_ratios,
            curvature,
            relative_density.unsqueeze(-1),
            normalized_radius.unsqueeze(-1),
            radial_distance.unsqueeze(-1),
        ),
        dim=-1,
    )

    normalized_centers = (centers - object_center) / object_scale.unsqueeze(-1)
    pairwise_distance = torch.cdist(normalized_centers, normalized_centers)
    neighbor_count = min(max(1, int(graph_k)), max(1, patch_count - 1))
    neighbor_indices = pairwise_distance.topk(
        k=min(patch_count, neighbor_count + 1), largest=False, dim=-1
    ).indices[..., 1:]
    if neighbor_indices.shape[-1] == 0:
        neighbor_indices = torch.zeros(
            batch_size, patch_count, 1, dtype=torch.long, device=points.device
        )
    batch = torch.arange(batch_size, device=points.device)[:, None, None]
    neighbor_normals = normals[batch, neighbor_indices]
    neighbor_curvature = curvature.squeeze(-1)[batch, neighbor_indices]
    neighbor_density = log_density[batch, neighbor_indices]
    neighbor_radius = normalized_radius[batch, neighbor_indices]
    center_normals = normals.unsqueeze(2)
    edge_distance = pairwise_distance.gather(2, neighbor_indices)
    normal_agreement = (center_normals * neighbor_normals).sum(dim=-1).abs()
    curvature_difference = (
        curvature.squeeze(-1).unsqueeze(2) - neighbor_curvature
    ).abs()
    density_log_ratio = (log_density.unsqueeze(2) - neighbor_density).abs()
    scale_log_ratio = (
        torch.log(normalized_radius.clamp_min(eps)).unsqueeze(2)
        - torch.log(neighbor_radius.clamp_min(eps))
    ).abs()
    edge_features = torch.stack(
        (
            edge_distance,
            normal_agreement,
            curvature_difference,
            density_log_ratio,
            scale_log_ratio,
        ),
        dim=-1,
    )
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "neighbor_indices": neighbor_indices,
        "centers": centers,
    }


def compute_patch_geometry_descriptor(points, patch_indices):
    """Compatibility alias returning the seven invariant node scalars."""
    return compute_invariant_patch_graph_inputs(points, patch_indices)["node_features"]


class InvariantGraphMessageBlock(nn.Module):
    def __init__(self, hidden_dim, edge_dim=EDGE_FEATURE_DIM):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, features, neighbor_indices, edge_features):
        batch_size = features.shape[0]
        batch = torch.arange(batch_size, device=features.device)[:, None, None]
        neighbors = features[batch, neighbor_indices]
        centers = features.unsqueeze(2).expand_as(neighbors)
        messages = self.message(torch.cat((centers, neighbors, edge_features), dim=-1))
        aggregated = messages.mean(dim=2)
        update = self.update(torch.cat((features, aggregated), dim=-1))
        return self.normalization(features + update)


class SE3InvariantPatchGraphEncoder(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=128, graph_k=8, num_layers=2):
        super().__init__()
        self.graph_k = int(graph_k)
        self.node_projection = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [InvariantGraphMessageBlock(hidden_dim) for _ in range(num_layers)]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, points, patch_indices):
        graph = compute_invariant_patch_graph_inputs(
            points, patch_indices, graph_k=self.graph_k
        )
        features = self.node_projection(graph["node_features"])
        for block in self.blocks:
            features = block(
                features, graph["neighbor_indices"], graph["edge_features"]
            )
        return self.output_projection(features)


class PointCloudAbnormalityPrior(nn.Module):
    """Generate a prompt-token prior using invariant graph routing only."""

    def __init__(
        self,
        feature_dim=1280,
        text_dim=1280,
        prompt_token_dim=1664,
        graph_dim=128,
        hidden_dim=512,
        graph_k=8,
        graph_layers=2,
        routing_temperature=0.2,
        top_m=10,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.top_m = int(top_m)
        self.routing_temperature = float(routing_temperature)
        self.geometry_encoder = SE3InvariantPatchGraphEncoder(
            hidden_dim=graph_dim,
            output_dim=graph_dim,
            graph_k=graph_k,
            num_layers=graph_layers,
        )
        self.geometry_router = nn.Linear(graph_dim, 1)
        self.network = nn.Sequential(
            nn.Linear(graph_dim + 2 * text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, prompt_token_dim),
            nn.Tanh(),
        )
        nn.init.zeros_(self.network[-2].weight)
        nn.init.zeros_(self.network[-2].bias)

    def forward(
        self,
        patch_feat,
        normal_text_embed,
        abnormal_text_proto,
        points,
        patch_indices,
    ):
        if patch_feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected patch feature dim {self.feature_dim}, got {patch_feat.shape[-1]}"
            )
        batch_size = patch_feat.shape[0]
        normal = normal_text_embed
        if normal.ndim == 1:
            normal = normal.unsqueeze(0)
        if normal.shape[0] == 1:
            normal = normal.expand(batch_size, -1)
        abnormal = abnormal_text_proto
        if abnormal.ndim == 1:
            abnormal = abnormal.unsqueeze(0)
        if abnormal.shape[0] == 1:
            abnormal = abnormal.expand(batch_size, -1)

        geometry = self.geometry_encoder(points, patch_indices)
        route_logits = self.geometry_router(geometry).squeeze(-1)
        route_weights = torch.softmax(
            route_logits / self.routing_temperature, dim=1
        )
        pooled_geometry = (geometry * route_weights.unsqueeze(-1)).sum(dim=1)
        prior = self.network(torch.cat((pooled_geometry, normal, abnormal), dim=-1))

        normalized_patch = F.normalize(patch_feat, dim=-1)
        initial_scores = (
            normalized_patch * F.normalize(abnormal, dim=-1)[:, None, :]
        ).sum(dim=-1) - (
            normalized_patch * F.normalize(normal, dim=-1)[:, None, :]
        ).sum(dim=-1)
        top_count = min(max(1, self.top_m), route_weights.shape[1])
        top_indices = torch.topk(route_weights, k=top_count, dim=1).indices
        return {
            "prior": prior,
            "top_indices": top_indices,
            "initial_patch_scores": initial_scores,
            "geometry_graph_feature": geometry,
            "route_logits": route_logits,
            "route_weights": route_weights,
            # Backward-compatible name used by existing evaluation/tests.
            "routing_weights": route_weights,
        }


def random_se3_transform(points, translation_scale=0.5):
    """Apply an independent random proper rotation and translation per sample."""
    batch_size = points.shape[0]
    random_matrix = torch.randn(
        batch_size, 3, 3, device=points.device, dtype=points.dtype
    )
    rotation, _ = torch.linalg.qr(random_matrix)
    determinant = torch.linalg.det(rotation)
    rotation[:, :, -1] *= torch.where(
        determinant < 0,
        rotation.new_tensor(-1.0),
        rotation.new_tensor(1.0),
    ).unsqueeze(-1)
    translation = translation_scale * torch.randn(
        batch_size, 1, 3, device=points.device, dtype=points.dtype
    )
    transformed_xyz = points[..., :3] @ rotation.transpose(1, 2) + translation
    if points.shape[-1] == 3:
        return transformed_xyz
    return torch.cat((transformed_xyz, points[..., 3:]), dim=-1)


def geometric_prior_invariance_loss(prior, transformed_prior):
    return F.mse_loss(prior, transformed_prior)


def normal_sample_prior_loss(prior, object_labels):
    normal_mask = object_labels <= 0
    if not normal_mask.any():
        return prior.sum() * 0.0
    return prior[normal_mask].square().mean()
