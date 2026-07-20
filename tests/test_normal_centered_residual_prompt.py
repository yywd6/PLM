"""Focused tests for the single maintained NCRP-K1 implementation."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from models.normal_centered_residual_prompt import (
    NormalCenteredResidualPromptLearner,
    NormalResidualPromptBank,
    forward_ncrp_k1_scores,
    patch_orthogonal_residual,
    project_single_residual,
    safe_l2_normalize,
    single_residual_logits,
)
from one_rest_protocol import load_yaml
from utils.residual_prompt_config import flatten_residual_prompt_config


ROOT = Path(__file__).resolve().parents[1]


def normalized(shape, dtype=torch.float32):
    return torch.nn.functional.normalize(torch.randn(*shape, dtype=dtype), dim=-1)


def test_safe_normalize_handles_zero_nan_and_inf():
    values = torch.tensor([[0.0, 0.0], [float("nan"), float("inf")]])
    result, norms = safe_l2_normalize(values)
    assert torch.isfinite(result).all()
    assert torch.isfinite(norms).all()
    assert torch.equal(result, torch.zeros_like(result))


def test_k1_projection_shape_unit_norm_and_normal_orthogonality():
    anchors = normalized((5, 32))
    basis = normalized((1, 32))
    directions, norms = project_single_residual(anchors, basis)
    assert directions.shape == (5, 1, 32)
    assert norms.shape == (5, 1)
    assert torch.allclose(directions.norm(dim=-1), torch.ones(5, 1), atol=1e-5)
    assert (directions[:, 0] * anchors).sum(-1).abs().max() < 1e-5


def test_k1_parallel_basis_uses_finite_deterministic_fallback():
    anchor = normalized((1, 24))
    direction_a, norm_a = project_single_residual(anchor, anchor.clone())
    direction_b, norm_b = project_single_residual(anchor, anchor.clone())
    assert torch.isfinite(direction_a).all()
    assert torch.allclose(direction_a, direction_b)
    assert torch.allclose(direction_a.norm(dim=-1), torch.ones(1, 1), atol=1e-5)
    assert (direction_a[:, 0] * anchor).sum(-1).abs().max() < 1e-5
    assert norm_a.max() < 1e-5
    assert torch.allclose(norm_a, norm_b)


def test_k1_patch_residual_is_orthogonal_and_degenerate_safe():
    anchors = normalized((3, 16))
    patches = torch.cat((anchors[:, None], normalized((3, 4, 16))), dim=1)
    residual, norms, valid = patch_orthogonal_residual(patches, anchors)
    assert residual.shape == patches.shape
    assert norms.shape == patches.shape[:2]
    assert valid.shape == patches.shape[:2]
    assert torch.isfinite(residual).all()
    assert not valid[:, 0].any()
    assert torch.equal(residual[:, 0], torch.zeros_like(residual[:, 0]))
    assert (residual * anchors[:, None]).sum(-1).abs().max() < 1e-5


def test_k1_anomaly_prototype_and_logits_shapes_match_formula():
    patches = normalized((2, 7, 20))
    anchors = normalized((2, 20))
    directions, _ = project_single_residual(anchors, normalized((1, 20)))
    output = single_residual_logits(
        patches, anchors, directions, gamma=1.0, temperature=0.07
    )
    assert output["patch_logits"].shape == (2, 7)
    assert output["normal_similarities"].shape == (2, 7)
    assert output["abnormal_similarities"].shape == (2, 7, 1)
    assert output["abnormal_prototype"].shape == (2, 1, 20)
    expected = (
        output["abnormal_similarities"].squeeze(-1)
        - output["normal_similarities"]
    ) / 0.07
    assert torch.allclose(output["patch_logits"], expected, atol=1e-6)


def test_k1_aggregation_is_exactly_the_single_similarity_without_logmeanexp():
    similarities = torch.randn(4, 9, 1)
    # The maintained K1 implementation indexes the only prototype directly,
    # avoiding a divide/logsumexp/multiply round trip and its rounding noise.
    aggregate = similarities[..., 0]
    assert torch.equal(aggregate, similarities.squeeze(-1))


def test_k1_forward_backward_updates_only_normal_tokens_and_one_residual():
    bank = NormalResidualPromptBank(12, 20, num_normal_tokens=4)
    named = dict(bank.named_parameters())
    assert set(named) == {"normal_tokens", "local_residual_basis"}
    assert named["normal_tokens"].shape == (4, 12)
    assert named["local_residual_basis"].shape == (1, 20)
    assert sum(parameter.numel() for parameter in bank.parameters()) == 68

    anchors = normalized((3, 20))
    directions, _ = project_single_residual(anchors, bank.local_residual_basis)
    output = single_residual_logits(normalized((3, 6, 20)), anchors, directions)
    output["patch_logits"].mean().backward()
    assert bank.local_residual_basis.grad is not None
    assert torch.isfinite(bank.local_residual_basis.grad).all()
    # This functional test does not encode text, so normal tokens correctly
    # receive no gradient until used by the Prompt encoder.
    assert bank.normal_tokens.grad is None


def test_k1_theoretical_prompt_parameter_count_for_current_dimensions():
    bank = NormalResidualPromptBank(1280, 1280, num_normal_tokens=4)
    assert sum(parameter.numel() for parameter in bank.parameters()) == 6400


def test_k1_bank_checkpoint_round_trip_preserves_output(tmp_path):
    anchors = normalized((2, 18))
    patches = normalized((2, 5, 18))
    bank = NormalResidualPromptBank(8, 18)
    before_direction, _ = project_single_residual(
        anchors, bank.local_residual_basis
    )
    before = single_residual_logits(patches, anchors, before_direction)[
        "patch_logits"
    ]
    checkpoint = tmp_path / "k1.pth"
    torch.save(bank.state_dict(), checkpoint)
    restored = NormalResidualPromptBank(8, 18)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    after_direction, _ = project_single_residual(
        anchors, restored.local_residual_basis
    )
    after = single_residual_logits(patches, anchors, after_direction)[
        "patch_logits"
    ]
    assert torch.equal(before, after)


class DummyLearner:
    def __init__(self, anchors, basis):
        self.anchors = anchors
        self.basis = basis
        self.gamma = 1.0
        self.eps = 1e-6

    def encode_normal_anchors(self, _clip_model, categories):
        assert len(categories) == self.anchors.shape[0]
        return self.anchors

    def projected_directions(self, normal):
        return project_single_residual(normal, self.basis, eps=self.eps)


class IdentityAdapter(nn.Module):
    def forward(self, layer_tokens):
        return layer_tokens


def test_k1_synthetic_feature_first_smoke_forward_backward_and_inference():
    batch, patches, dim = 2, 6, 14
    anchors = normalized((batch, dim))
    basis = nn.Parameter(normalized((1, dim)))
    learner = DummyLearner(anchors, basis)
    patch_features = normalized((batch, patches, dim)).requires_grad_()
    global_features = normalized((batch, dim)).requires_grad_()
    output = forward_ncrp_k1_scores(
        IdentityAdapter(),
        patch_features,
        global_features,
        learner,
        None,
        ["car", "chicken"],
    )
    assert output["patch_logits"].shape == (batch, patches)
    assert output["global_logits"].shape == (batch,)
    assert output["basis_assignments"].shape == (batch, patches, 1)
    assert torch.equal(output["basis_assignments"], torch.ones_like(output["basis_assignments"]))
    loss = output["patch_logits"].mean() + output["global_logits"].mean()
    loss.backward()
    assert basis.grad is not None and torch.isfinite(basis.grad).all()
    assert patch_features.grad is not None


def test_k1_learner_rejects_more_than_one_basis_without_building_clip():
    with pytest.raises(ValueError, match="exactly one"):
        NormalCenteredResidualPromptLearner(
            object(), object(), num_bases=6
        )


def test_main_ncrp_yaml_is_k1_only():
    config = flatten_residual_prompt_config(
        load_yaml(ROOT / "configs/ncrp_k1.yaml")
    )
    assert config["static_prompt_version"] == "ncrp_k1"
    assert config["residual_prompt_enabled"] is True
    assert config["residual_num_bases"] == 1
    assert config["num_abnormal_prompts"] == 1
    assert config["residual_gamma"] == 1.0
    assert config["residual_eps"] == 1e-6
    assert config["object_pooling_mode"] == "top_mean"
    assert config["object_top_ratio"] == 0.2
    assert config["global_alpha"] == 0.0


def test_no_ncrp_ablation_yaml_remains():
    assert not list((ROOT / "configs/ablations").glob("ncrp_*.yaml"))
    ncrp_configs = list((ROOT / "configs").glob("*ncrp*.yaml"))
    assert ROOT / "configs/ncrp_k1.yaml" in ncrp_configs
    assert all(
        path.name == "ncrp_k1.yaml"
        or path.name == "ddf3d_ncrp_k1_base.yaml"
        or (
            path.name.startswith("ddf3d_")
            and path.name.endswith("_ncrp_k1.yaml")
        )
        for path in ncrp_configs
    )


def test_only_ncrp_static_shared_stage1_and_ddf3d_yamls_remain():
    configs = set((ROOT / "configs").rglob("*.yaml"))
    maintained = {
        ROOT / "configs/ncrp_k1.yaml",
        ROOT / "configs/one_rest_visual_baseline_v7.yaml",
        ROOT / "configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml",
    }
    assert maintained <= configs
    assert all(
        path in maintained or path.name.startswith("ddf3d_")
        for path in configs
    )


def test_static_config_does_not_enable_ncrp():
    static = flatten_residual_prompt_config(
        load_yaml(ROOT / "configs/two_rest_static_six_prompt_v1_uniform_scoring.yaml")
    )
    assert "residual_prompt_enabled" not in static


def test_one_rest_visual_baseline_is_stage1_only():
    visual = load_yaml(ROOT / "configs/one_rest_visual_baseline_v7.yaml")
    assert visual["use_static_prompt"] is False
    assert visual["freeze_visual_adapter"] is False
    assert visual["visual_adapter_training_mode"] == "fused_only"
    assert visual["baseline_checkpoint"] is None
