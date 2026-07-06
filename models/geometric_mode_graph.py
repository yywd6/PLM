"""SE(3)-invariant patch graph for geometric mode prompting."""

import torch
import torch.nn as nn


NODE_FEATURE_DIM = 7
EDGE_FEATURE_DIM = 5


def compute_se3_invariant_patch_graph(points, patch_indices, graph_k=8, eps=1e-6):
    """Build a patch kNN graph using only rotation/translation invariant scalars."""
    xyz = points[..., :3]
    indices = patch_indices.long()
    batch_size, patch_count, neighborhood_size = indices.shape
    batch = torch.arange(batch_size, device=points.device)[:, None, None]
    patches = xyz[batch, indices]
    centers = patches.mean(dim=2)
    centered = patches - centers.unsqueeze(2)

    covariance = centered.transpose(-1, -2) @ centered
    covariance = covariance / max(1, neighborhood_size)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    eigenvalue_ratios = eigenvalues / eigenvalues.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
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
    edge_features = torch.stack(
        (
            pairwise_distance.gather(2, neighbor_indices),
            (normals.unsqueeze(2) * neighbor_normals).sum(dim=-1).abs(),
            (curvature.squeeze(-1).unsqueeze(2) - neighbor_curvature).abs(),
            (log_density.unsqueeze(2) - neighbor_density).abs(),
            (
                torch.log(normalized_radius.clamp_min(eps)).unsqueeze(2)
                - torch.log(neighbor_radius.clamp_min(eps))
            ).abs(),
        ),
        dim=-1,
    )
    return node_features, edge_features, neighbor_indices


class _InvariantMessageBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + EDGE_FEATURE_DIM, hidden_dim),
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
    """Encode invariant node/edge scalars without using oriented coordinates."""

    def __init__(self, hidden_dim=128, output_dim=128, graph_k=8, num_layers=2):
        super().__init__()
        self.graph_k = int(graph_k)
        self.node_projection = nn.Sequential(
            nn.Linear(NODE_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_InvariantMessageBlock(hidden_dim) for _ in range(int(num_layers))]
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, points, patch_indices):
        nodes, edges, neighbors = compute_se3_invariant_patch_graph(
            points, patch_indices, self.graph_k
        )
        features = self.node_projection(nodes)
        for block in self.blocks:
            features = block(features, neighbors, edges)
        return self.output_projection(features)


def random_se3_transform(points, translation_scale=0.5):
    batch_size = points.shape[0]
    matrix = torch.randn(batch_size, 3, 3, device=points.device, dtype=points.dtype)
    rotation, _ = torch.linalg.qr(matrix)
    determinant = torch.linalg.det(rotation)
    rotation[:, :, -1] *= torch.where(
        determinant < 0, rotation.new_tensor(-1.0), rotation.new_tensor(1.0)
    ).unsqueeze(-1)
    translation = translation_scale * torch.randn(
        batch_size, 1, 3, device=points.device, dtype=points.dtype
    )
    transformed_xyz = points[..., :3] @ rotation.transpose(1, 2) + translation
    if points.shape[-1] == 3:
        return transformed_xyz
    return torch.cat((transformed_xyz, points[..., 3:]), dim=-1)
