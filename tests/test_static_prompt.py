import torch
import torch.nn as nn

from models.static_prompt import (
    StaticPromptLearner,
    encode_static_prompts,
    prompt_diversity_loss,
    static_prompt_anomaly_logits,
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
        self.positional_embedding = nn.Parameter(
            torch.zeros(context_length, width)
        )
        self.text_projection = nn.Parameter(torch.randn(width, output_dim))
        self.attn_mask = None
        self.text_pool_type = "argmax"


def tokenizer(texts):
    tokens = torch.zeros(len(texts), 24, dtype=torch.long)
    tokens[:, 0] = 1
    tokens[:, 1:5] = torch.tensor([2, 3, 4, 5])
    tokens[:, 5] = 63
    return tokens


def build_learner(num_prompts=6):
    clip = DummyClip()
    learner = StaticPromptLearner(
        clip,
        tokenizer,
        num_prompts=num_prompts,
        num_normal_tokens=2,
        num_abnormal_tokens=2,
    )
    return clip, learner


def test_static_prompts_are_finite_and_trainable():
    clip, learner = build_learner()
    prompts = encode_static_prompts(learner, clip, ["car", "airplane"])
    patch_embeddings = torch.randn(2, 8, 6)
    logits = static_prompt_anomaly_logits(
        patch_embeddings,
        prompts["normal_text_embed"],
        prompts["abnormal_text_embeds"],
    )
    assert prompts["normal_text_embed"].shape == (2, 6)
    assert prompts["abnormal_text_embeds"].shape == (2, 6, 6)
    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()
    logits.mean().backward()
    assert learner.prompt_bank.normal_tokens.grad is not None
    assert learner.prompt_bank.abnormal_tokens.grad is not None
    assert torch.isfinite(learner.prompt_bank.abnormal_tokens.grad).all()


def test_uniform_log_mean_exp_scoring_matches_manual_formula():
    patch = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    normal = torch.tensor([[0.0, 1.0]])
    abnormal = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    score_temperature = 0.1
    logits = static_prompt_anomaly_logits(
        patch,
        normal,
        abnormal,
        temperature=1.0,
        prompt_score_temperature=score_temperature,
    )
    patches = torch.nn.functional.normalize(patch, dim=-1)
    prompts = torch.nn.functional.normalize(abnormal, dim=-1)
    similarities = torch.einsum("bgd,bkd->bgk", patches, prompts)
    expected_abnormal = score_temperature * (
        torch.logsumexp(similarities / score_temperature, dim=-1)
        - torch.log(torch.tensor(2.0))
    )
    expected_normal = (
        patches * torch.nn.functional.normalize(normal, dim=-1)[:, None]
    ).sum(dim=-1)
    assert torch.allclose(logits, expected_abnormal - expected_normal)


def test_prompt_diversity_penalizes_duplicate_embeddings():
    distinct = torch.eye(3).unsqueeze(0)
    duplicate = torch.ones(1, 3, 3)
    assert prompt_diversity_loss(distinct) < prompt_diversity_loss(duplicate)
