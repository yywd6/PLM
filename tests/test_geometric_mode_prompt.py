import inspect

import torch
import torch.nn as nn

from models.geometric_mode_graph import random_se3_transform
from models.geometric_mode_prompt import (
    GeometricModePromptLearner,
    encode_geometric_mode_prompts,
    geometry_gate_supervision_loss,
    mode_aware_anomaly_logits,
    mode_entropy_regularization,
    point_mask_to_patch_targets,
    sinkhorn_mode_assignment_loss,
)


class DummyTransformer(nn.Module):
    def get_cast_dtype(self):
        return torch.float32

    def forward(self, x, attn_mask=None):
        return x


class DummyClip(nn.Module):
    def __init__(self, width=8, output_dim=6, context_length=24):
        super().__init__()
        self.token_embedding = nn.Embedding(64, width)
        self.transformer = DummyTransformer()
        self.ln_final = nn.LayerNorm(width)
        self.positional_embedding = nn.Parameter(torch.zeros(context_length, width))
        self.text_projection = nn.Parameter(torch.randn(width, output_dim))
        self.attn_mask = None
        self.text_pool_type = "argmax"


def tokenizer(texts):
    tokens = torch.zeros(len(texts), 24, dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, 1:5] = torch.tensor([2, 3, 4, 5])
    tokens[:, 5] = 63
    return tokens


def geometry_batch(batch_size=2, points=64, patches=8, patch_size=8):
    xyz = torch.randn(batch_size, points, 3)
    indices = torch.randint(0, points, (batch_size, patches, patch_size))
    return xyz, indices


def build_learner():
    clip = DummyClip()
    learner = GeometricModePromptLearner(
        clip,
        tokenizer,
        num_modes=3,
        num_normal_tokens=2,
        num_abnormal_tokens=2,
        graph_dim=16,
        graph_k=3,
        graph_layers=2,
        modulator_hidden_dim=16,
    )
    nn.init.normal_(learner.prompt_modulator.output_projection.weight, std=0.01)
    return clip, learner


def test_mode_prompt_shapes_probabilities_and_finite_outputs():
    clip, learner = build_learner()
    points, indices = geometry_batch()
    geometry = learner.forward_geometry(points, indices)
    prompts = encode_geometric_mode_prompts(
        learner, clip, ["car", "airplane"], geometry["delta_A"]
    )

    assert prompts["normal_text_embed"].shape == (2, 6)
    assert prompts["dynamic_abnormal_text_embeds"].shape == (2, 3, 6)
    assert geometry["mode_weights"].shape == (2, 3)
    assert geometry["node_mode_weights"].shape == (2, 8, 3)
    assert geometry["node_mode_logits"].shape == (2, 8, 3)
    assert geometry["abnormal_gate_logits"].shape == (2, 8)
    assert geometry["delta_A"].shape == (2, 3, 2, 8)
    assert torch.allclose(geometry["mode_weights"].sum(dim=1), torch.ones(2))
    assert torch.allclose(
        geometry["node_mode_weights"].sum(dim=-1), torch.ones(2, 8)
    )
    for value in (*prompts.values(), geometry["mode_weights"], geometry["delta_A"]):
        assert torch.isfinite(value).all()


def test_mode_routing_and_residual_are_se3_invariant():
    torch.manual_seed(13)
    _, learner = build_learner()
    points, indices = geometry_batch()
    original = learner.forward_geometry(points, indices)
    transformed = learner.forward_geometry(random_se3_transform(points), indices)
    assert torch.allclose(
        original["mode_weights"], transformed["mode_weights"], atol=2e-4, rtol=2e-4
    )
    assert torch.allclose(
        original["delta_A"], transformed["delta_A"], atol=2e-4, rtol=2e-4
    )
    assert torch.allclose(
        original["abnormal_gate_logits"], transformed["abnormal_gate_logits"],
        atol=2e-4, rtol=2e-4,
    )


def test_mode_entropy_rejects_fixed_uniform_and_collapsed_assignments():
    uniform = torch.full((3, 3), 1.0 / 3.0)
    collapsed = torch.tensor([[1.0, 0.0, 0.0]]).expand(3, -1)
    balanced_confident = torch.eye(3)
    uniform_loss = mode_entropy_regularization(uniform)
    collapsed_loss = mode_entropy_regularization(collapsed)
    balanced_loss = mode_entropy_regularization(balanced_confident)
    assert torch.allclose(uniform_loss, torch.tensor(0.5), atol=1e-6)
    assert torch.allclose(collapsed_loss, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(balanced_loss, torch.tensor(0.0), atol=1e-6)
    assert balanced_loss < uniform_loss < collapsed_loss

    # Patch-node routing uses [B, G, K]; leading dimensions must be pooled,
    # never accumulated as additional entropy categories.
    uniform_nodes = uniform[:, None, :].expand(-1, 8, -1)
    assert torch.allclose(
        mode_entropy_regularization(uniform_nodes), torch.tensor(0.5), atol=1e-6
    )


def test_logsumexp_scoring_and_geometry_gate_are_finite():
    patch = torch.tensor([[[1.0, 0.0], [0.5, 0.5]]])
    normal = torch.tensor([[0.0, 1.0]])
    abnormal = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    routing = torch.tensor([[[0.7, 0.3], [0.2, 0.8]]])
    gate = torch.tensor([[0.25, -0.5]])
    without_gate = mode_aware_anomaly_logits(
        patch, normal, abnormal, routing, temperature=1.0,
        mode_score_type="logsumexp", mode_score_temperature=0.1,
    )
    with_gate = mode_aware_anomaly_logits(
        patch, normal, abnormal, routing, temperature=1.0,
        mode_score_type="logsumexp", mode_score_temperature=0.1,
        abnormal_gate_logits=gate, gate_logit_scale=2.0,
    )
    assert torch.isfinite(with_gate).all()
    assert torch.allclose(with_gate - without_gate, 2.0 * gate)


def test_balanced_geometry_gate_prefers_correct_patch_ordering():
    targets = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    correct = torch.tensor([[2.0, 1.0, -1.0, -2.0]])
    reversed_logits = -correct
    assert geometry_gate_supervision_loss(correct, targets) < (
        geometry_gate_supervision_loss(reversed_logits, targets)
    )


def test_sinkhorn_mode_assignment_is_finite_and_trainable():
    logits = torch.randn(2, 8, 3, requires_grad=True)
    anomaly_mask = torch.tensor(
        [[True, False, True, False, True, False, True, False],
         [False, True, False, True, False, True, False, True]]
    )
    loss = sinkhorn_mode_assignment_loss(
        logits, anomaly_mask, epsilon=0.05, iterations=3,
        prediction_temperature=0.2,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_geometry_never_modifies_or_concatenates_visual_patch_features():
    clip, learner = build_learner()
    points, indices = geometry_batch()
    geometry = learner.forward_geometry(points, indices)
    prompts = encode_geometric_mode_prompts(
        learner, clip, ["car", "car"], geometry["delta_A"]
    )
    visual = torch.randn(2, 8, 6)
    unchanged = visual.clone()
    logits = mode_aware_anomaly_logits(
        visual,
        prompts["normal_text_embed"],
        prompts["dynamic_abnormal_text_embeds"],
        geometry["node_mode_weights"],
    )
    assert logits.shape == (2, 8)
    assert torch.equal(visual, unchanged)
    assert "patch" not in inspect.signature(learner.forward_geometry).parameters


def test_patch_mode_weights_change_patch_scores():
    patch = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    normal = torch.tensor([[0.0, 1.0]])
    abnormal = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    routing = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    logits = mode_aware_anomaly_logits(
        patch, normal, abnormal, routing, temperature=1.0
    )
    assert torch.allclose(logits, torch.tensor([[1.0, -1.0]]))


def test_anomaly_patch_mask_selects_only_positive_source_patches():
    labels = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    indices = torch.tensor([[[0, 3], [1, 2], [0, 1]]])
    targets, ratios = point_mask_to_patch_targets(labels, indices, threshold=0.25)
    assert torch.equal(targets, torch.tensor([[False, True, True]]))
    assert torch.allclose(ratios, torch.tensor([[0.0, 1.0, 0.5]]))


def test_mode_entropy_selection_ignores_normal_patches():
    weights = torch.tensor([[[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]])
    anomaly_mask = torch.tensor([[True, False, True]])
    selected = mode_entropy_regularization(weights, 0.5, anomaly_mask)
    assert torch.allclose(selected, torch.tensor(0.0), atol=1e-6)
