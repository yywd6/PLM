import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometric_cap import (
    GeometricCompoundPromptLearner,
    encode_geometric_prompts,
    encode_prior_enabled_abnormal_prompts,
    geometric_anomaly_logits,
)
from models.geometric_dap import (
    PointCloudAbnormalityPrior,
    compute_patch_geometry_descriptor,
    random_se3_transform,
)
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


def make_geometry_batch(batch_size=2, point_count=64, patch_count=8, patch_size=8):
    points = torch.randn(batch_size, point_count, 3)
    indices = torch.randint(0, point_count, (batch_size, patch_count, patch_size))
    return points, indices


def test_cap_outputs_normal_and_k_abnormal_embeddings():
    clip = DummyClip()
    encoded_texts = []

    def recording_tokenizer(texts):
        encoded_texts.extend(texts)
        return tokenizer(texts)

    learner = GeometricCompoundPromptLearner(clip, recording_tokenizer, 5, 2, 3)
    outputs = encode_geometric_prompts(
        learner, clip, object_names=["car", "airplane"]
    )
    assert encoded_texts[:2] == [
        "a point cloud patch of a car",
        "a point cloud patch of an airplane",
    ]
    assert outputs["normal_text_embed"].shape == (2, 6)
    assert outputs["abnormal_text_embeds"].shape == (2, 5, 6)
    assert outputs["abnormal_text_proto"].shape == (2, 6)
    assert torch.isfinite(outputs["abnormal_text_embeds"]).all()


def test_dap_soft_routes_graph_and_produces_dynamic_prompts():
    clip = DummyClip()
    learner = GeometricCompoundPromptLearner(clip, tokenizer, 4, 2, 2)
    base = encode_geometric_prompts(learner, clip)
    points, indices = make_geometry_batch()
    patch_feat = torch.randn(2, 8, 6)
    dap = PointCloudAbnormalityPrior(
        6, 6, learner.token_width, graph_dim=16, hidden_dim=16,
        graph_k=3, graph_layers=2, top_m=3,
    )
    result = dap(
        patch_feat, base["normal_text_embed"], base["abnormal_text_proto"],
        points, indices,
    )
    assert result["top_indices"].shape == (2, 3)
    assert result["prior"].shape == (2, learner.token_width)
    assert result["geometry_graph_feature"].shape == (2, 8, 16)
    assert result["route_logits"].shape == (2, 8)
    assert result["route_weights"].shape == (2, 8)
    assert torch.equal(result["route_weights"], result["routing_weights"])
    assert torch.allclose(result["routing_weights"].sum(dim=1), torch.ones(2))
    dynamic = encode_geometric_prompts(
        learner, clip, result["prior"]
    )
    dynamic_only = encode_prior_enabled_abnormal_prompts(
        learner, clip, result["prior"]
    )
    assert dynamic["prior_enabled_abnormal_text_embeds"].shape == (2, 4, 6)
    assert torch.allclose(
        dynamic_only["prior_enabled_abnormal_text_proto"],
        dynamic["prior_enabled_abnormal_text_proto"],
    )
    assert torch.isfinite(dynamic["prior_enabled_abnormal_text_proto"]).all()


def test_geometry_descriptor_contains_seven_invariant_node_scalars():
    points, indices = make_geometry_batch(patch_count=5)
    descriptor = compute_patch_geometry_descriptor(points, indices)
    assert descriptor.shape == (2, 5, 7)
    assert torch.isfinite(descriptor).all()


def test_graph_prior_is_rotation_translation_invariant():
    torch.manual_seed(7)
    clip = DummyClip()
    learner = GeometricCompoundPromptLearner(clip, tokenizer, 4, 2, 2)
    base = encode_geometric_prompts(learner, clip)
    points, indices = make_geometry_batch()
    patch_feat = torch.randn(2, 8, 6)
    dap = PointCloudAbnormalityPrior(
        6, 6, learner.token_width, graph_dim=16, hidden_dim=16,
        graph_k=3, graph_layers=2, routing_temperature=0.3, top_m=3,
    )
    nn.init.normal_(dap.network[3].weight, std=0.01)
    nn.init.normal_(dap.network[3].bias, std=0.01)
    original = dap(
        patch_feat, base["normal_text_embed"], base["abnormal_text_proto"],
        points, indices,
    )
    transformed = dap(
        patch_feat, base["normal_text_embed"], base["abnormal_text_proto"],
        random_se3_transform(points), indices,
    )
    assert torch.allclose(
        original["geometry_graph_feature"],
        transformed["geometry_graph_feature"],
        atol=2e-4,
        rtol=2e-4,
    )
    assert torch.allclose(
        original["routing_weights"], transformed["routing_weights"],
        atol=2e-4, rtol=2e-4,
    )
    cosine = F.cosine_similarity(original["prior"], transformed["prior"], dim=-1)
    assert torch.all(cosine > 0.9999)



def test_cap_disabled_score_matches_original_baseline():
    patches = torch.randn(2, 5, 6)
    normal = torch.randn(1, 6)
    abnormal = torch.randn(1, 6)
    original = patch_text_logits(patches, normal, abnormal, 0.2)
    geometric = geometric_anomaly_logits(patches, normal, abnormal.squeeze(0), None, 0.2)
    assert torch.allclose(original, geometric, atol=1e-6)
