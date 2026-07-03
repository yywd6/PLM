import torch
import torch.nn as nn

from models.geometric_cap import (
    GeometricCompoundPromptLearner,
    encode_geometric_prompts,
    geometric_anomaly_logits,
)
from models.geometric_dap import PointCloudAbnormalityPrior, compute_patch_geometry_descriptor
from models.trainable_baseline import patch_text_logits


class DummyTransformer(nn.Module):
    def get_cast_dtype(self):
        return torch.float32

    def forward(self, x, attn_mask=None):
        return x


class DummyClip(nn.Module):
    def __init__(self, width=8, output_dim=6, context_length=12):
        super().__init__()
        self.token_embedding = nn.Embedding(32, width)
        self.transformer = DummyTransformer()
        self.ln_final = nn.LayerNorm(width)
        self.positional_embedding = nn.Parameter(torch.zeros(context_length, width))
        self.text_projection = nn.Parameter(torch.randn(width, output_dim))
        self.attn_mask = None
        self.text_pool_type = "argmax"


def tokenizer(texts):
    tokens = torch.zeros(len(texts), 12, dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, 1:4] = torch.tensor([2, 3, 4])
    tokens[:, 4] = 31
    return tokens


def test_cap_outputs_normal_and_k_abnormal_embeddings():
    clip = DummyClip()
    learner = GeometricCompoundPromptLearner(clip, tokenizer, 5, 2, 3)
    outputs = encode_geometric_prompts(learner, clip)
    assert outputs["normal_text_embed"].shape == (1, 6)
    assert outputs["abnormal_text_embeds"].shape == (5, 6)
    assert outputs["abnormal_text_proto"].shape == (6,)
    assert torch.isfinite(outputs["abnormal_text_embeds"]).all()


def test_dap_selects_top_m_and_produces_dynamic_prompts():
    clip = DummyClip()
    learner = GeometricCompoundPromptLearner(clip, tokenizer, 4, 2, 2)
    base = encode_geometric_prompts(learner, clip)
    patch_feat = torch.randn(2, 7, 6)
    geometry = torch.randn(2, 7, 4)
    dap = PointCloudAbnormalityPrior(6, 6, learner.token_width, 4, 16, top_m=3)
    result = dap(
        patch_feat, base["normal_text_embed"], base["abnormal_text_proto"], geometry
    )
    assert result["top_indices"].shape == (2, 3)
    assert result["prior"].shape == (2, learner.token_width)
    dynamic = encode_geometric_prompts(learner, clip, result["prior"])
    assert dynamic["prior_enabled_abnormal_text_embeds"].shape == (2, 4, 6)
    assert torch.isfinite(dynamic["prior_enabled_abnormal_text_proto"]).all()


def test_geometry_descriptor_is_prompt_side_shape_only():
    points = torch.randn(2, 8, 3)
    indices = torch.tensor([[[0, 1, 2], [3, 4, 5]]] * 2)
    descriptor = compute_patch_geometry_descriptor(points, indices)
    assert descriptor.shape == (2, 2, 4)
    assert torch.isfinite(descriptor).all()


def test_cap_disabled_score_matches_original_baseline():
    patches = torch.randn(2, 5, 6)
    normal = torch.randn(1, 6)
    abnormal = torch.randn(1, 6)
    original = patch_text_logits(patches, normal, abnormal, 0.2)
    geometric = geometric_anomaly_logits(patches, normal, abnormal.squeeze(0), None, 0.2)
    assert torch.allclose(original, geometric, atol=1e-6)
