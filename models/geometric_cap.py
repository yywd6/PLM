"""Fine-grained learnable 3D geometric compound prompts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricCompoundPromptLearner(nn.Module):
    """Shared normal tokens plus K abnormal-specific token groups.

    Prompts use the category-conditioned ``a point cloud patch of {article} {category}`` suffix.
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        num_abnormal_prompts=10,
        num_normal_tokens=4,
        num_abnormal_tokens=4,
        suffix="a point cloud patch of {article} {category}",
    ):
        super().__init__()
        if num_abnormal_prompts < 1:
            raise ValueError("num_abnormal_prompts must be positive")
        self.num_abnormal_prompts = int(num_abnormal_prompts)
        self.num_normal_tokens = int(num_normal_tokens)
        self.num_abnormal_tokens = int(num_abnormal_tokens)
        self.suffix_template = str(suffix)
        self.default_object_name = "object"
        self.tokenizer = tokenizer
        token_width = clip_model.token_embedding.weight.shape[1]
        self.token_width = int(token_width)

        self.normal_tokens = nn.Parameter(torch.empty(self.num_normal_tokens, token_width))
        self.abnormal_tokens = nn.Parameter(
            torch.empty(self.num_abnormal_prompts, self.num_abnormal_tokens, token_width)
        )
        nn.init.normal_(self.normal_tokens, std=0.02)
        nn.init.normal_(self.abnormal_tokens, std=0.02)

    def _normalize_object_names(self, object_names, batch_size=None):
        if object_names is None:
            names = [self.default_object_name] * (batch_size or 1)
        elif isinstance(object_names, str):
            names = [object_names]
        else:
            names = [str(name) for name in object_names]
        if not names or any(not name.strip() for name in names):
            raise ValueError("object_names must contain non-empty category names")
        if batch_size is not None and len(names) != batch_size:
            raise ValueError(
                f"object_names has {len(names)} entries, expected batch size {batch_size}"
            )
        return names

    def _assemble(self, context, clip_model, object_names):
        if context.ndim != 3:
            raise ValueError(f"context must be [P, T, C], got {tuple(context.shape)}")
        prompt_count, context_length, _ = context.shape
        names = self._normalize_object_names(object_names, prompt_count)
        try:
            suffixes = [
                self.suffix_template.format(
                    category=name,
                    article="an" if name[0].lower() in "aeiou" else "a",
                )
                for name in names
            ]
        except (KeyError, ValueError) as error:
            raise ValueError(
                "geometric prompt suffix must be a valid template with {article} and {category}"
            ) from error
        tokenized_suffix = self.tokenizer(suffixes).long()
        if tokenized_suffix.ndim == 1:
            tokenized_suffix = tokenized_suffix.unsqueeze(0)
        tokenized_suffix = tokenized_suffix.to(context.device)
        with torch.no_grad():
            embedded = clip_model.token_embedding(tokenized_suffix).to(context.dtype)

        sequence_length = tokenized_suffix.shape[1]
        suffix_length = sequence_length - 1 - context_length
        if suffix_length <= 0:
            raise ValueError("Learnable tokens leave no room for the fixed suffix")
        prefix = embedded[:, :1]
        suffix = embedded[:, 1 : 1 + suffix_length]
        prompts = torch.cat((prefix, context, suffix), dim=1)

        # Move category-conditioned suffix tokens and EOT after the learnable context.
        tokenized = torch.zeros_like(tokenized_suffix)
        tokenized[:, :1] = tokenized_suffix[:, :1]
        tokenized[:, 1 + context_length :] = tokenized_suffix[:, 1 : 1 + suffix_length]
        return prompts, tokenized

    def normal_prompt(self, clip_model, object_names=None):
        names = self._normalize_object_names(object_names)
        context = self.normal_tokens.unsqueeze(0).expand(len(names), -1, -1)
        return self._assemble(context, clip_model, names)

    def abnormal_prompts(self, clip_model, object_names=None, prior=None):
        if prior is not None:
            if prior.ndim != 2 or prior.shape[1] != self.token_width:
                raise ValueError(
                    f"prior must be [B, {self.token_width}], got {tuple(prior.shape)}"
                )
            names = self._normalize_object_names(object_names, prior.shape[0])
        else:
            names = self._normalize_object_names(object_names)

        batch_size = len(names)
        shared = self.normal_tokens[None, None].expand(
            batch_size, self.num_abnormal_prompts, -1, -1
        )
        abnormal = self.abnormal_tokens.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if prior is not None:
            abnormal = abnormal + prior[:, None, None, :]
        context = torch.cat((shared, abnormal), dim=2).reshape(
            batch_size * self.num_abnormal_prompts,
            self.num_normal_tokens + self.num_abnormal_tokens,
            self.token_width,
        )
        repeated_names = [name for name in names for _ in range(self.num_abnormal_prompts)]
        return self._assemble(context, clip_model, repeated_names)


def _encode_prompt_embeddings(clip_model, prompt_embeddings, tokenized_prompts):
    cast_dtype = clip_model.transformer.get_cast_dtype()
    sequence_length = prompt_embeddings.shape[1]
    x = prompt_embeddings.to(device=clip_model.positional_embedding.device, dtype=cast_dtype)
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


def encode_prior_enabled_abnormal_prompts(
    prompt_learner, clip_model, prior, object_names=None
):
    """Encode only prior-conditioned abnormal prompts, without base recomputation."""
    if prior is None:
        raise ValueError("prior is required for dynamic abnormal prompts")
    names = prompt_learner._normalize_object_names(object_names, prior.shape[0])
    batch_size = len(names)
    dynamic_prompt, dynamic_tokens = prompt_learner.abnormal_prompts(
        clip_model, names, prior
    )
    dynamic = _encode_prompt_embeddings(
        clip_model, dynamic_prompt, dynamic_tokens
    ).reshape(batch_size, prompt_learner.num_abnormal_prompts, -1)
    return {
        "prior_enabled_abnormal_text_embeds": dynamic,
        "prior_enabled_abnormal_text_proto": F.normalize(
            dynamic.mean(dim=1), dim=-1
        ),
    }


def encode_geometric_prompts(
    prompt_learner, clip_model, prior=None, object_names=None
):
    """Encode category-conditioned prompts as [B,D], [B,K,D], and [B,D]."""
    if prior is not None:
        names = prompt_learner._normalize_object_names(object_names, prior.shape[0])
    else:
        names = prompt_learner._normalize_object_names(object_names)
    batch_size = len(names)

    normal_prompt, normal_tokens = prompt_learner.normal_prompt(clip_model, names)
    abnormal_prompt, abnormal_tokens = prompt_learner.abnormal_prompts(clip_model, names)
    normal_embedding = _encode_prompt_embeddings(clip_model, normal_prompt, normal_tokens)
    abnormal_embeddings = _encode_prompt_embeddings(
        clip_model, abnormal_prompt, abnormal_tokens
    ).reshape(batch_size, prompt_learner.num_abnormal_prompts, -1)
    outputs = {
        "normal_text_embed": normal_embedding,
        "abnormal_text_embeds": abnormal_embeddings,
        "abnormal_text_proto": F.normalize(abnormal_embeddings.mean(dim=1), dim=-1),
        "prior_enabled_abnormal_text_embeds": None,
        "prior_enabled_abnormal_text_proto": None,
    }
    if prior is not None:
        outputs.update(
            encode_prior_enabled_abnormal_prompts(
                prompt_learner, clip_model, prior, object_names=names
            )
        )
    return outputs


def abnormal_prompt_orthogonal_loss(abnormal_embeddings):
    prompt_count = abnormal_embeddings.shape[-2]
    if prompt_count <= 1:
        return abnormal_embeddings.sum() * 0.0
    embeddings = F.normalize(abnormal_embeddings, dim=-1)
    gram = embeddings @ embeddings.transpose(-1, -2)
    identity = torch.eye(prompt_count, device=gram.device, dtype=gram.dtype)
    batch_factor = gram.numel() // (prompt_count * prompt_count)
    off_diagonal_count = batch_factor * prompt_count * (prompt_count - 1)
    return ((gram - identity) ** 2).sum() / off_diagonal_count


def geometric_anomaly_logits(
    patch_embeddings,
    normal_text_embed,
    abnormal_text_proto,
    prior_enabled_abnormal_text_proto=None,
    temperature=0.07,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    patch_embeddings = F.normalize(patch_embeddings, dim=-1)
    normal = F.normalize(normal_text_embed, dim=-1)
    if normal.shape[0] == 1:
        normal_similarity = torch.matmul(patch_embeddings, normal.T).squeeze(-1)
    else:
        normal_similarity = (patch_embeddings * normal[:, None, :]).sum(dim=-1)
    base = F.normalize(abnormal_text_proto, dim=-1)
    if base.ndim == 1:
        base_similarity = torch.matmul(patch_embeddings, base)
    else:
        base_similarity = (patch_embeddings * base[:, None, :]).sum(dim=-1)
    abnormal_similarity = base_similarity
    if prior_enabled_abnormal_text_proto is not None:
        dynamic = F.normalize(prior_enabled_abnormal_text_proto, dim=-1)
        dynamic_similarity = (patch_embeddings * dynamic[:, None, :]).sum(dim=-1)
        abnormal_similarity = 0.5 * base_similarity + 0.5 * dynamic_similarity
    return (abnormal_similarity - normal_similarity) / temperature
