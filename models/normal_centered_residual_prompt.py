"""NCRP-K1: normal-centered prompting with one shared residual vector.

This is the only maintained NCRP implementation.  The frozen text encoder
produces a category-aware normal anchor.  One learnable residual vector is
projected into that anchor's orthogonal complement and added to the normal
anchor to construct the abnormal prototype.  There is no QR, adaptive router,
cover/reject loss, basis diversity loss, or dual branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.static_prompt import (
    _encode_prompt_embeddings,
    format_category_prompt,
)


def safe_l2_normalize(value, dim=-1, eps=1e-6):
    """Normalize in float32 and return finite zeros for degenerate vectors."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    finite = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.vector_norm(finite, dim=dim, keepdim=True)
    normalized = finite / norm.clamp_min(float(eps))
    normalized = torch.where(
        norm > float(eps), normalized, torch.zeros_like(normalized)
    )
    return normalized.to(dtype=value.dtype), norm.squeeze(dim)


def _orthogonal_fallback(normal_anchor, eps):
    """Build a deterministic unit direction orthogonal to each anchor."""
    normal, _ = safe_l2_normalize(normal_anchor, eps=eps)
    coordinate_index = normal.abs().argmin(dim=-1, keepdim=True)
    coordinate = torch.zeros_like(normal).scatter_(-1, coordinate_index, 1.0)
    direction = coordinate - (
        (coordinate * normal).sum(dim=-1, keepdim=True) * normal
    )
    direction, _ = safe_l2_normalize(direction, eps=eps)
    return direction


def project_single_residual(normal_anchor, residual_vector, eps=1e-6):
    """Project ``[1,D]`` residual vector into every ``[B,D]`` normal complement."""
    if normal_anchor.ndim != 2:
        raise ValueError("normal_anchor must have shape [B,D]")
    if residual_vector.ndim != 2 or residual_vector.shape[0] != 1:
        raise ValueError("NCRP-K1 residual_vector must have shape [1,D]")
    if normal_anchor.shape[-1] != residual_vector.shape[-1]:
        raise ValueError("normal anchor and residual vector dimensions differ")

    normal, _ = safe_l2_normalize(normal_anchor.float(), eps=eps)
    residual = torch.nan_to_num(
        residual_vector.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    raw = residual.unsqueeze(0) - torch.einsum(
        "kd,bd->bk", residual, normal
    ).unsqueeze(-1) * normal.unsqueeze(1)
    direction, norm = safe_l2_normalize(raw, eps=eps)
    fallback = _orthogonal_fallback(normal, eps).unsqueeze(1)
    direction = torch.where((norm > eps).unsqueeze(-1), direction, fallback)
    direction, _ = safe_l2_normalize(direction, eps=eps)
    return torch.nan_to_num(direction), norm


def patch_orthogonal_residual(patch_embeddings, normal_anchor, eps=1e-6):
    """Return the component of every patch orthogonal to its normal anchor."""
    patches, _ = safe_l2_normalize(patch_embeddings.float(), eps=eps)
    normal, _ = safe_l2_normalize(normal_anchor.float(), eps=eps)
    raw = patches - torch.einsum(
        "bgd,bd->bg", patches, normal
    ).unsqueeze(-1) * normal.unsqueeze(1)
    residual, norm = safe_l2_normalize(raw, eps=eps)
    valid = norm > float(eps)
    residual = torch.where(
        valid.unsqueeze(-1), residual, torch.zeros_like(residual)
    )
    return residual, norm, valid


def single_residual_logits(
    patch_embeddings,
    normal_anchor,
    residual_direction,
    gamma=1.0,
    temperature=0.07,
    eps=1e-6,
):
    """Score patches against one normal-centered abnormal prototype."""
    if residual_direction.ndim != 3 or residual_direction.shape[1] != 1:
        raise ValueError("residual_direction must have shape [B,1,D]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    patches, _ = safe_l2_normalize(patch_embeddings.float(), eps=eps)
    normal, _ = safe_l2_normalize(normal_anchor.float(), eps=eps)
    abnormal, _ = safe_l2_normalize(
        normal + float(gamma) * residual_direction[:, 0].float(), eps=eps
    )
    normal_similarity = torch.einsum("bgd,bd->bg", patches, normal)
    abnormal_similarity = torch.einsum("bgd,bd->bg", patches, abnormal)
    return {
        "patch_logits": (
            abnormal_similarity - normal_similarity
        ) / float(temperature),
        "normal_similarities": normal_similarity,
        "abnormal_similarities": abnormal_similarity.unsqueeze(-1),
        "abnormal_prototype": abnormal.unsqueeze(1),
    }


def _text_embedding_dim(clip_model):
    projection = getattr(clip_model, "text_projection", None)
    if isinstance(projection, nn.Linear):
        return int(projection.out_features)
    if projection is not None and hasattr(projection, "shape"):
        return int(projection.shape[-1])
    raise ValueError("Cannot infer frozen text encoder output dimension")


class NormalResidualPromptBank(nn.Module):
    """Four normal context tokens and exactly one residual vector."""

    def __init__(self, token_dim, embedding_dim, num_normal_tokens=4):
        super().__init__()
        self.num_bases = 1
        self.num_normal_tokens = int(num_normal_tokens)
        self.token_dim = int(token_dim)
        self.embedding_dim = int(embedding_dim)
        self.normal_tokens = nn.Parameter(
            torch.empty(self.num_normal_tokens, self.token_dim)
        )
        # Keep the historical state-dict key for existing K1 checkpoints.
        self.local_residual_basis = nn.Parameter(
            torch.empty(1, self.embedding_dim)
        )
        nn.init.normal_(self.normal_tokens, std=0.02)
        nn.init.normal_(self.local_residual_basis, std=0.02)
        with torch.no_grad():
            self.local_residual_basis.copy_(
                F.normalize(self.local_residual_basis, dim=-1)
            )

    @property
    def residual_basis(self):
        return self.local_residual_basis


class NormalCenteredResidualPromptLearner(nn.Module):
    """Category-aware normal Prompt plus the shared NCRP-K1 residual."""

    def __init__(
        self,
        clip_model,
        tokenizer,
        num_bases=1,
        num_normal_tokens=4,
        prompt_template="a point cloud patch of a {category}",
        use_category_prompt=True,
        gamma=1.0,
        eps=1e-6,
    ):
        super().__init__()
        if int(num_bases) != 1:
            raise ValueError("NCRP supports exactly one residual vector")
        if gamma < 0 or eps <= 0:
            raise ValueError("gamma must be non-negative and eps positive")
        token_dim = int(clip_model.token_embedding.weight.shape[1])
        embedding_dim = _text_embedding_dim(clip_model)
        self.num_bases = 1
        self.num_prompts = 1
        self.embedding_dim = embedding_dim
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.prompt_template = str(prompt_template)
        self.use_category_prompt = bool(use_category_prompt)
        self.tokenizer = tokenizer
        self.prompt_bank = NormalResidualPromptBank(
            token_dim,
            embedding_dim,
            num_normal_tokens=num_normal_tokens,
        )

    def prompt_texts(self, categories):
        return [
            format_category_prompt(
                self.prompt_template, category, self.use_category_prompt
            )
            for category in categories
        ]

    def normal_prompt(self, clip_model, categories):
        context = self.prompt_bank.normal_tokens.unsqueeze(0).expand(
            len(categories), -1, -1
        )
        tokenized_suffix = self.tokenizer(self.prompt_texts(categories)).long()
        if tokenized_suffix.ndim == 1:
            tokenized_suffix = tokenized_suffix.unsqueeze(0)
        tokenized_suffix = tokenized_suffix.to(context.device)
        with torch.no_grad():
            embedded = clip_model.token_embedding(tokenized_suffix).to(
                context.dtype
            )
        suffix_length = tokenized_suffix.shape[1] - 1 - context.shape[1]
        if suffix_length <= 0:
            raise ValueError("Learnable tokens leave no room for category suffix")
        prompts = torch.cat(
            (embedded[:, :1], context, embedded[:, 1 : 1 + suffix_length]),
            dim=1,
        )
        tokenized = torch.zeros_like(tokenized_suffix)
        tokenized[:, :1] = tokenized_suffix[:, :1]
        tokenized[:, 1 + context.shape[1] :] = tokenized_suffix[
            :, 1 : 1 + suffix_length
        ]
        return prompts, tokenized

    def encode_normal_anchors(self, clip_model, categories):
        prompts, tokens = self.normal_prompt(clip_model, categories)
        return _encode_prompt_embeddings(clip_model, prompts, tokens)

    def projected_directions(self, normal_anchor):
        return project_single_residual(
            normal_anchor,
            self.prompt_bank.local_residual_basis,
            eps=self.eps,
        )


def forward_ncrp_k1_scores(
    adapter,
    layer_tokens,
    global_embeddings,
    learner,
    clip_model,
    categories,
    temperature=0.07,
    patch_centers=None,
):
    """Feature-first NCRP-K1 forward used by training and inference."""
    normal = learner.encode_normal_anchors(clip_model, categories)
    directions, projection_norms = learner.projected_directions(normal)
    from models.ddf3d import is_ddf3d_adapter

    ddf3d_output = None
    if is_ddf3d_adapter(adapter):
        feature_output = adapter.forward_features(layer_tokens)
        patches = feature_output["patch_embeddings"]
        patch_output = single_residual_logits(
            patches,
            normal,
            directions,
            gamma=learner.gamma,
            temperature=temperature,
            eps=learner.eps,
        )
        layer_margins = torch.stack(
            [
                single_residual_logits(
                    layer,
                    normal,
                    directions,
                    gamma=learner.gamma,
                    temperature=temperature,
                    eps=learner.eps,
                )["patch_logits"]
                for layer in feature_output["projected_layers"]
            ],
            dim=-1,
        )
        ddf3d_output = adapter.enhance_scores(
            patch_output["patch_logits"],
            layer_margins,
            feature_output,
            patch_centers,
        )
        patch_output = {**patch_output, **ddf3d_output}
    else:
        patches = adapter(layer_tokens)
        patch_output = single_residual_logits(
            patches,
            normal,
            directions,
            gamma=learner.gamma,
            temperature=temperature,
            eps=learner.eps,
        )
    global_output = single_residual_logits(
        global_embeddings.unsqueeze(1),
        normal,
        directions,
        gamma=learner.gamma,
        temperature=temperature,
        eps=learner.eps,
    )
    patch_residual, residual_norms, valid_residual = patch_orthogonal_residual(
        patches, normal, eps=learner.eps
    )
    basis_similarities = torch.einsum(
        "bgd,bkd->bgk", patch_residual, directions
    )
    assignments = torch.ones_like(basis_similarities)
    combined_norm = torch.linalg.vector_norm(
        directions[:, 0], dim=-1, keepdim=True
    ).unsqueeze(1).expand(-1, patches.shape[1], -1)
    normal_inner = torch.einsum("bkd,bd->bk", directions, normal).abs().max(-1).values
    zeros = normal_inner.new_zeros(normal_inner.shape)
    output = {
        **patch_output,
        "basis_similarities": basis_similarities,
        "basis_assignments": assignments,
        "patch_residual": patch_residual,
        "residual_norms": residual_norms,
        "valid_residual": valid_residual,
        "combined_residual_direction_norm": combined_norm,
        "patch_embeddings": patches,
        "global_logits": global_output["patch_logits"].squeeze(1),
        "normal_text_embed": normal,
        "projected_directions": directions,
        "projection_norms": projection_norms,
        "diversity_embeddings": directions,
        "basis_normal_max_abs_inner_product": normal_inner,
        "orthogonalized_basis_gram_off_diagonal_mean": zeros,
    }
    if ddf3d_output is not None:
        output["ddf3d"] = ddf3d_output
    return output
