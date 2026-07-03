"""Sample-wise point-cloud abnormality prior for dynamic abnormal prompts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_patch_geometry_descriptor(points, patch_indices):
    """Return local eigenvalue ratios and RMS radius; never modifies visual features."""
    descriptors = []
    xyz = points[..., :3]
    for batch_index in range(points.shape[0]):
        patches = xyz[batch_index][patch_indices[batch_index].long()]
        centered = patches - patches.mean(dim=1, keepdim=True)
        covariance = centered.transpose(1, 2) @ centered
        covariance = covariance / max(1, patches.shape[1])
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        ratios = eigenvalues / eigenvalues.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        radius = centered.square().sum(dim=-1).mean(dim=-1, keepdim=True).sqrt()
        radius = radius / radius.mean().clamp_min(1e-6)
        descriptors.append(torch.cat((ratios, radius), dim=-1))
    return torch.stack(descriptors, dim=0)


class PointCloudAbnormalityPrior(nn.Module):
    def __init__(
        self,
        feature_dim=1280,
        text_dim=1280,
        prompt_token_dim=1664,
        geo_dim=4,
        hidden_dim=512,
        top_m=10,
    ):
        super().__init__()
        self.top_m = int(top_m)
        self.geo_dim = int(geo_dim)
        input_dim = feature_dim + 2 * text_dim + self.geo_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
        patch_geo_desc=None,
    ):
        patch_feat = F.normalize(patch_feat, dim=-1)
        batch_size, patch_count, _ = patch_feat.shape
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
        initial_scores = (
            patch_feat * F.normalize(abnormal, dim=-1)[:, None, :]
        ).sum(dim=-1) - (
            patch_feat * F.normalize(normal, dim=-1)[:, None, :]
        ).sum(dim=-1)
        top_count = min(max(1, self.top_m), patch_count)
        top_indices = torch.topk(initial_scores, k=top_count, dim=1).indices
        feature_indices = top_indices.unsqueeze(-1).expand(-1, -1, patch_feat.shape[-1])
        pooled_feature = torch.gather(patch_feat, 1, feature_indices).mean(dim=1)
        if patch_geo_desc is None:
            pooled_geometry = patch_feat.new_zeros(batch_size, self.geo_dim)
        else:
            if patch_geo_desc.shape[:2] != patch_feat.shape[:2]:
                raise ValueError("patch_geo_desc and patch_feat must share [B, G]")
            if patch_geo_desc.shape[-1] != self.geo_dim:
                raise ValueError(
                    f"Expected geo_dim={self.geo_dim}, got {patch_geo_desc.shape[-1]}"
                )
            geo_indices = top_indices.unsqueeze(-1).expand(
                -1, -1, patch_geo_desc.shape[-1]
            )
            pooled_geometry = torch.gather(
                patch_geo_desc, 1, geo_indices
            ).mean(dim=1)
        prior_input = torch.cat((pooled_feature, normal, abnormal, pooled_geometry), dim=-1)
        prior = self.network(prior_input)
        return {
            "prior": prior,
            "top_indices": top_indices,
            "initial_patch_scores": initial_scores,
        }


def normal_sample_prior_loss(prior, object_labels):
    normal_mask = object_labels <= 0
    if not normal_mask.any():
        return prior.sum() * 0.0
    return prior[normal_mask].square().mean()
