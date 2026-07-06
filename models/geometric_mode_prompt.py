"""SE(3)-Invariant Geometric Mode Prompting."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometric_mode_graph import SE3InvariantPatchGraphEncoder
from models.geometric_mode_router import GeometricModeRouter, ModeSpecificPromptModulator


def format_category_prompt(template, category, use_category_prompt=True):
    """Format a category suffix while preserving the existing a/an behavior."""
    if not use_category_prompt:
        return "point cloud patch"
    category = str(category).strip()
    if not category:
        raise ValueError("category must be non-empty")
    article = "an" if category[0].lower() in "aeiou" else "a"
    try:
        text = template.format(category=category, article=article)
    except (KeyError, ValueError) as error:
        raise ValueError("prompt_template must support {category} and optional {article}") from error
    # The requested default contains a literal 'a {category}'. Correct it for vowels.
    if article == "an" and "{article}" not in template:
        suffix = f" a {category}"
        if text.endswith(suffix):
            text = text[: -len(suffix)] + f" an {category}"
    return text


class GeometricModePromptBank(nn.Module):
    """Shared normal tokens and K independent abnormal mode token groups."""

    def __init__(self, token_dim, num_modes=10, num_normal_tokens=4, num_abnormal_tokens=4):
        super().__init__()
        self.num_modes = int(num_modes)
        self.num_normal_tokens = int(num_normal_tokens)
        self.num_abnormal_tokens = int(num_abnormal_tokens)
        self.token_dim = int(token_dim)
        self.normal_tokens = nn.Parameter(
            torch.empty(self.num_normal_tokens, self.token_dim)
        )
        self.abnormal_tokens = nn.Parameter(
            torch.empty(self.num_modes, self.num_abnormal_tokens, self.token_dim)
        )
        nn.init.normal_(self.normal_tokens, std=0.02)
        nn.init.normal_(self.abnormal_tokens, std=0.02)


class GeometricModePromptLearner(nn.Module):
    """Category-aware prompt bank conditioned by K-specific invariant geometry modes."""

    def __init__(
        self,
        clip_model,
        tokenizer,
        num_modes=10,
        num_normal_tokens=4,
        num_abnormal_tokens=4,
        prompt_template="a point cloud patch of a {category}",
        use_category_prompt=True,
        graph_dim=128,
        graph_k=8,
        graph_layers=2,
        router_temperature=0.2,
        residual_scale=0.1,
        modulator_hidden_dim=512,
        use_mode_specific_residual=True,
    ):
        super().__init__()
        token_dim = int(clip_model.token_embedding.weight.shape[1])
        self.num_modes = int(num_modes)
        self.prompt_template = str(prompt_template)
        self.use_category_prompt = bool(use_category_prompt)
        self.use_mode_specific_residual = bool(use_mode_specific_residual)
        self.tokenizer = tokenizer
        self.prompt_bank = GeometricModePromptBank(
            token_dim, num_modes, num_normal_tokens, num_abnormal_tokens
        )
        self.graph_encoder = SE3InvariantPatchGraphEncoder(
            hidden_dim=graph_dim,
            output_dim=graph_dim,
            graph_k=graph_k,
            num_layers=graph_layers,
        )
        self.mode_router = GeometricModeRouter(
            graph_dim=graph_dim,
            num_modes=num_modes,
            temperature=router_temperature,
        )
        self.prompt_modulator = ModeSpecificPromptModulator(
            graph_dim=graph_dim,
            num_modes=num_modes,
            num_abnormal_tokens=num_abnormal_tokens,
            token_dim=token_dim,
            hidden_dim=modulator_hidden_dim,
            residual_scale=residual_scale,
        )

    @property
    def token_dim(self):
        return self.prompt_bank.token_dim

    @property
    def num_abnormal_tokens(self):
        return self.prompt_bank.num_abnormal_tokens

    def prompt_texts(self, categories):
        return [
            format_category_prompt(
                self.prompt_template, category, self.use_category_prompt
            )
            for category in categories
        ]

    def forward_geometry(self, points, patch_indices):
        graph_features = self.graph_encoder(points, patch_indices)
        routed = self.mode_router(graph_features)
        if self.use_mode_specific_residual:
            delta_a = self.prompt_modulator(routed["mode_features"])
        else:
            delta_a = graph_features.new_zeros(
                graph_features.shape[0],
                self.num_modes,
                self.num_abnormal_tokens,
                self.token_dim,
            )
        return {"graph_features": graph_features, "delta_A": delta_a, **routed}

    def _assemble(self, context, clip_model, texts):
        if context.ndim != 3:
            raise ValueError(f"context must be [P,T,C], got {tuple(context.shape)}")
        prompt_count, context_length, _ = context.shape
        if len(texts) != prompt_count:
            raise ValueError(f"Expected {prompt_count} prompt texts, got {len(texts)}")
        tokenized_suffix = self.tokenizer(texts).long()
        if tokenized_suffix.ndim == 1:
            tokenized_suffix = tokenized_suffix.unsqueeze(0)
        tokenized_suffix = tokenized_suffix.to(context.device)
        with torch.no_grad():
            embedded = clip_model.token_embedding(tokenized_suffix).to(context.dtype)
        suffix_length = tokenized_suffix.shape[1] - 1 - context_length
        if suffix_length <= 0:
            raise ValueError("Learnable tokens leave no room for the category suffix")
        prompts = torch.cat(
            (embedded[:, :1], context, embedded[:, 1 : 1 + suffix_length]), dim=1
        )
        tokenized = torch.zeros_like(tokenized_suffix)
        tokenized[:, :1] = tokenized_suffix[:, :1]
        tokenized[:, 1 + context_length :] = tokenized_suffix[:, 1 : 1 + suffix_length]
        return prompts, tokenized

    def normal_prompt(self, clip_model, categories):
        texts = self.prompt_texts(categories)
        context = self.prompt_bank.normal_tokens.unsqueeze(0).expand(
            len(categories), -1, -1
        )
        return self._assemble(context, clip_model, texts)

    def abnormal_mode_prompts(self, clip_model, categories, delta_a=None):
        batch_size = len(categories)
        shared = self.prompt_bank.normal_tokens[None, None].expand(
            batch_size, self.num_modes, -1, -1
        )
        abnormal = self.prompt_bank.abnormal_tokens.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        if delta_a is not None:
            expected = (
                batch_size,
                self.num_modes,
                self.num_abnormal_tokens,
                self.token_dim,
            )
            if tuple(delta_a.shape) != expected:
                raise ValueError(f"delta_A must be {expected}, got {tuple(delta_a.shape)}")
            abnormal = abnormal + delta_a
        context = torch.cat((shared, abnormal), dim=2).reshape(
            batch_size * self.num_modes,
            self.prompt_bank.num_normal_tokens + self.num_abnormal_tokens,
            self.token_dim,
        )
        texts = [text for text in self.prompt_texts(categories) for _ in range(self.num_modes)]
        return self._assemble(context, clip_model, texts)


def _encode_prompt_embeddings(clip_model, prompt_embeddings, tokenized_prompts):
    cast_dtype = clip_model.transformer.get_cast_dtype()
    sequence_length = prompt_embeddings.shape[1]
    x = prompt_embeddings.to(
        device=clip_model.positional_embedding.device, dtype=cast_dtype
    )
    x = x + clip_model.positional_embedding[:sequence_length].to(dtype=cast_dtype)
    attention_mask = clip_model.attn_mask
    if attention_mask is not None:
        attention_mask = attention_mask[:sequence_length, :sequence_length]
    x = clip_model.transformer(x, attn_mask=attention_mask)
    x = clip_model.ln_final(x)
    tokens = tokenized_prompts.to(x.device)
    pool_type = getattr(clip_model, "text_pool_type", "argmax")
    eos_token_id = getattr(clip_model, "text_eos_id", None)
    if pool_type == "first":
        x = x[:, 0]
    elif pool_type == "last":
        x = x[:, -1]
    elif pool_type == "eos" and eos_token_id is not None:
        indices = (tokens == eos_token_id).int().argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), indices]
    else:
        x = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(dim=-1)]
    projection = clip_model.text_projection
    if isinstance(projection, nn.Linear):
        x = projection(x)
    elif projection is not None:
        x = x @ projection
    return F.normalize(x.float(), dim=-1)


def encode_geometric_mode_prompts(learner, clip_model, categories, delta_a):
    """Return normal [B,D], base modes [B,K,D], and dynamic modes [B,K,D]."""
    batch_size = len(categories)
    normal_prompt, normal_tokens = learner.normal_prompt(clip_model, categories)
    base_prompt, base_tokens = learner.abnormal_mode_prompts(
        clip_model, categories, delta_a=None
    )
    dynamic_prompt, dynamic_tokens = learner.abnormal_mode_prompts(
        clip_model, categories, delta_a=delta_a
    )
    normal = _encode_prompt_embeddings(clip_model, normal_prompt, normal_tokens)
    base = _encode_prompt_embeddings(clip_model, base_prompt, base_tokens).reshape(
        batch_size, learner.num_modes, -1
    )
    dynamic = _encode_prompt_embeddings(
        clip_model, dynamic_prompt, dynamic_tokens
    ).reshape(batch_size, learner.num_modes, -1)
    return {
        "normal_text_embed": normal,
        "base_abnormal_text_embeds": base,
        "dynamic_abnormal_text_embeds": dynamic,
    }


def mode_aware_anomaly_logits(
    patch_embeddings,
    normal_text_embed,
    dynamic_abnormal_text_embeds,
    mode_weights,
    temperature=0.07,
    mode_score_type="weighted_sum",
    use_mode_weighted_scoring=True,
    mode_score_temperature=0.1,
    abnormal_gate_logits=None,
    gate_logit_scale=1.0,
):
    """Score every patch against every mode; never fuse geometry into visual features."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    patches = F.normalize(patch_embeddings, dim=-1)
    normal = F.normalize(normal_text_embed, dim=-1)
    abnormal = F.normalize(dynamic_abnormal_text_embeds, dim=-1)
    sim_modes = torch.einsum("bgd,bkd->bgk", patches, abnormal)
    if mode_weights.ndim == 2:
        weights = mode_weights[:, None, :]
    elif mode_weights.ndim == 3:
        if mode_weights.shape[:2] != sim_modes.shape[:2]:
            raise ValueError(
                "Patch mode weights must have shape [B,G,K] matching patch embeddings"
            )
        weights = mode_weights
    else:
        raise ValueError("mode_weights must have shape [B,K] or [B,G,K]")
    if weights.shape[-1] != sim_modes.shape[-1]:
        raise ValueError("Mode count does not match abnormal text embeddings")
    if not use_mode_weighted_scoring:
        weights = torch.full_like(weights, 1.0 / weights.shape[-1])
    if mode_score_type == "weighted_sum":
        abnormal_score = (sim_modes * weights).sum(dim=-1)
    elif mode_score_type == "logsumexp":
        if mode_score_temperature <= 0:
            raise ValueError("mode_score_temperature must be positive")
        abnormal_score = mode_score_temperature * torch.logsumexp(
            sim_modes / mode_score_temperature
            + torch.log(weights.clamp_min(1e-8)),
            dim=-1,
        )
    elif mode_score_type == "max":
        abnormal_score = sim_modes.max(dim=-1).values
    else:
        raise ValueError(f"Unsupported mode_score_type: {mode_score_type}")
    normal_score = (patches * normal[:, None, :]).sum(dim=-1)
    logits = (abnormal_score - normal_score) / temperature
    if abnormal_gate_logits is not None:
        gate_logits = abnormal_gate_logits
        if gate_logits.ndim == 1:
            gate_logits = gate_logits[:, None]
        if gate_logits.shape != logits.shape:
            raise ValueError("abnormal_gate_logits must match [B,G] anomaly logits")
        logits = logits + gate_logit_scale * gate_logits
    return logits


def mode_diversity_loss(abnormal_text_embeddings):
    mode_count = abnormal_text_embeddings.shape[-2]
    if mode_count <= 1:
        return abnormal_text_embeddings.sum() * 0.0
    embeddings = F.normalize(abnormal_text_embeddings, dim=-1)
    gram = embeddings @ embeddings.transpose(-1, -2)
    identity = torch.eye(mode_count, device=gram.device, dtype=gram.dtype)
    batch_factor = gram.numel() // (mode_count * mode_count)
    return ((gram - identity) ** 2).sum() / (
        batch_factor * mode_count * (mode_count - 1)
    )


def sinkhorn_mode_assignment_loss(
    node_mode_logits, selection_mask, epsilon=0.05, iterations=3,
    prediction_temperature=0.2,
):
    """Balanced self-labeling over source-category anomalous graph patches."""
    if epsilon <= 0 or prediction_temperature <= 0:
        raise ValueError("Sinkhorn temperatures must be positive")
    selected = node_mode_logits.reshape(-1, node_mode_logits.shape[-1])[
        selection_mask.reshape(-1).bool()
    ]
    if selected.shape[0] == 0:
        return node_mode_logits.sum() * 0.0
    with torch.no_grad():
        scores = selected.detach().float()
        scores = scores - scores.max(dim=-1, keepdim=True).values
        assignments = torch.exp(scores / epsilon).transpose(0, 1)
        assignments = assignments / assignments.sum().clamp_min(1e-8)
        mode_count, sample_count = assignments.shape
        for _ in range(max(1, int(iterations))):
            assignments = assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-8)
            assignments = assignments / mode_count
            assignments = assignments / assignments.sum(dim=0, keepdim=True).clamp_min(1e-8)
            assignments = assignments / sample_count
        targets = (assignments * sample_count).transpose(0, 1)
    log_probabilities = F.log_softmax(
        selected / prediction_temperature, dim=-1
    )
    return -(targets.to(log_probabilities.dtype) * log_probabilities).sum(dim=-1).mean()


def mode_entropy_regularization(
    mode_weights, conditional_entropy_weight=0.5, selection_mask=None, eps=1e-8
):
    """Stable anti-collapse objective: load balance plus conditional confidence.

    (1 - H(mode)) penalizes global single-mode collapse. The conditional term
    discourages a fixed uniform router. Balanced confident assignments reach 0.
    """
    log_modes = math.log(mode_weights.shape[-1])
    flat_weights = mode_weights.reshape(-1, mode_weights.shape[-1])
    if selection_mask is not None:
        flat_mask = selection_mask.reshape(-1).bool()
        if flat_mask.numel() != flat_weights.shape[0]:
            raise ValueError("selection_mask must match mode_weights leading dimensions")
        flat_weights = flat_weights[flat_mask]
        if flat_weights.shape[0] == 0:
            return mode_weights.sum() * 0.0
    conditional_entropy = -(
        flat_weights * torch.log(flat_weights.clamp_min(eps))
    ).sum(dim=-1).mean() / log_modes
    usage = flat_weights.mean(dim=0)
    marginal_entropy = -(
        usage * torch.log(usage.clamp_min(eps))
    ).sum() / log_modes
    return (1.0 - marginal_entropy) + conditional_entropy_weight * conditional_entropy


def normalized_mode_entropy(mode_weights, eps=1e-8):
    entropy = -(mode_weights * torch.log(mode_weights.clamp_min(eps))).sum(dim=-1)
    return entropy / math.log(mode_weights.shape[-1])


def geometry_gate_supervision_loss(gate_logits, patch_targets, margin=0.5, eps=1e-6):
    """Class-balanced source-mask supervision for the invariant abnormal gate."""
    targets = patch_targets.to(gate_logits.dtype)
    positives = targets.sum()
    negatives = targets.numel() - positives
    if positives <= 0 or negatives <= 0:
        return F.binary_cross_entropy_with_logits(gate_logits, targets)
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    balanced_bce = F.binary_cross_entropy_with_logits(
        gate_logits, targets, pos_weight=positive_weight
    )
    probabilities = gate_logits.sigmoid()
    dice = 1.0 - (
        2.0 * (probabilities * targets).sum() + eps
    ) / (probabilities.sum() + targets.sum() + eps)
    positive_mean = gate_logits[patch_targets.bool()].mean()
    negative_mean = gate_logits[~patch_targets.bool()].mean()
    ranking = F.relu(gate_logits.new_tensor(margin) - positive_mean + negative_mean)
    return balanced_bce + dice + ranking


def residual_suppression_loss(delta_a, object_labels):
    normal_mask = object_labels <= 0
    if not normal_mask.any():
        return delta_a.sum() * 0.0
    # Vector L2 keeps this meaningful for high-dimensional prompt tokens.
    return delta_a[normal_mask].square().sum(dim=-1).mean()


def point_mask_to_patch_targets(point_labels, patch_indices, threshold=0.05):
    """Convert source point masks to patch anomaly targets and ratios."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("patch anomaly threshold must be in [0, 1]")
    if point_labels.ndim != 2 or patch_indices.ndim != 3:
        raise ValueError("Expected point_labels [B,N] and patch_indices [B,G,M]")
    if point_labels.shape[0] != patch_indices.shape[0]:
        raise ValueError("Batch size mismatch between labels and patch indices")
    flat_indices = patch_indices.long().reshape(point_labels.shape[0], -1)
    patch_labels = torch.gather(point_labels.float(), 1, flat_indices).reshape(
        patch_indices.shape
    )
    ratios = patch_labels.mean(dim=-1)
    return ratios > threshold, ratios


def se3_sanity_loss(mode_weights, transformed_weights, delta_a, transformed_delta_a):
    return F.mse_loss(mode_weights, transformed_weights) + F.mse_loss(
        delta_a, transformed_delta_a
    )
