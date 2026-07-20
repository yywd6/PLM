"""Static learnable normal and abnormal prompts for 3D anomaly detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        raise ValueError(
            "prompt_template must support {category} and optional {article}"
        ) from error
    if article == "an" and "{article}" not in template:
        suffix = f" a {category}"
        if text.endswith(suffix):
            text = text[: -len(suffix)] + f" an {category}"
    return text


class StaticPromptBank(nn.Module):
    """One normal token group and K independent abnormal token groups."""

    def __init__(
        self, token_dim, num_prompts=6, num_normal_tokens=4, num_abnormal_tokens=4
    ):
        super().__init__()
        self.num_prompts = int(num_prompts)
        self.num_normal_tokens = int(num_normal_tokens)
        self.num_abnormal_tokens = int(num_abnormal_tokens)
        self.token_dim = int(token_dim)
        self.normal_tokens = nn.Parameter(
            torch.empty(self.num_normal_tokens, self.token_dim)
        )
        self.abnormal_tokens = nn.Parameter(
            torch.empty(
                self.num_prompts, self.num_abnormal_tokens, self.token_dim
            )
        )
        nn.init.normal_(self.normal_tokens, std=0.02)
        nn.init.normal_(self.abnormal_tokens, std=0.02)


class StaticPromptLearner(nn.Module):
    """A category-aware normal Prompt and K static abnormal Prompts."""

    def __init__(
        self,
        clip_model,
        tokenizer,
        num_prompts=6,
        num_normal_tokens=4,
        num_abnormal_tokens=4,
        prompt_template="a point cloud patch of a {category}",
        use_category_prompt=True,
    ):
        super().__init__()
        token_dim = int(clip_model.token_embedding.weight.shape[1])
        self.num_prompts = int(num_prompts)
        self.prompt_template = str(prompt_template)
        self.use_category_prompt = bool(use_category_prompt)
        self.tokenizer = tokenizer
        self.prompt_bank = StaticPromptBank(
            token_dim,
            num_prompts,
            num_normal_tokens,
            num_abnormal_tokens,
        )

    def prompt_texts(self, categories):
        return [
            format_category_prompt(
                self.prompt_template, category, self.use_category_prompt
            )
            for category in categories
        ]

    def _assemble(self, context, clip_model, texts):
        if context.ndim != 3:
            raise ValueError(f"context must be [P,T,C], got {tuple(context.shape)}")
        prompt_count, context_length, _ = context.shape
        if len(texts) != prompt_count:
            raise ValueError(
                f"Expected {prompt_count} prompt texts, got {len(texts)}"
            )
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
            (embedded[:, :1], context, embedded[:, 1 : 1 + suffix_length]),
            dim=1,
        )
        tokenized = torch.zeros_like(tokenized_suffix)
        tokenized[:, :1] = tokenized_suffix[:, :1]
        tokenized[:, 1 + context_length :] = tokenized_suffix[
            :, 1 : 1 + suffix_length
        ]
        return prompts, tokenized

    def normal_prompt(self, clip_model, categories):
        texts = self.prompt_texts(categories)
        context = self.prompt_bank.normal_tokens.unsqueeze(0).expand(
            len(categories), -1, -1
        )
        return self._assemble(context, clip_model, texts)

    def abnormal_prompts(self, clip_model, categories):
        batch_size = len(categories)
        shared_normal = self.prompt_bank.normal_tokens[None, None].expand(
            batch_size, self.num_prompts, -1, -1
        )
        abnormal = self.prompt_bank.abnormal_tokens.unsqueeze(0).expand(
            batch_size, -1, -1, -1
        )
        context = torch.cat((shared_normal, abnormal), dim=2).reshape(
            batch_size * self.num_prompts,
            self.prompt_bank.num_normal_tokens
            + self.prompt_bank.num_abnormal_tokens,
            self.prompt_bank.token_dim,
        )
        texts = [
            text
            for text in self.prompt_texts(categories)
            for _ in range(self.num_prompts)
        ]
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


def encode_static_prompts(learner, clip_model, categories):
    """Encode the normal Prompt and K abnormal Prompts once per batch."""
    batch_size = len(categories)
    normal_prompt, normal_tokens = learner.normal_prompt(clip_model, categories)
    abnormal_prompt, abnormal_tokens = learner.abnormal_prompts(
        clip_model, categories
    )
    normal = _encode_prompt_embeddings(
        clip_model, normal_prompt, normal_tokens
    )
    abnormal = _encode_prompt_embeddings(
        clip_model, abnormal_prompt, abnormal_tokens
    ).reshape(batch_size, learner.num_prompts, -1)
    return {
        "normal_text_embed": normal,
        "abnormal_text_embeds": abnormal,
    }


def static_prompt_anomaly_logits(
    patch_embeddings,
    normal_text_embed,
    abnormal_text_embeds,
    temperature=0.07,
    prompt_score_temperature=0.07,
):
    """Compute anomaly logits using uniform log-mean-exp Prompt aggregation."""
    if temperature <= 0 or prompt_score_temperature <= 0:
        raise ValueError("temperatures must be positive")
    patches = F.normalize(patch_embeddings, dim=-1)
    normal = F.normalize(normal_text_embed, dim=-1)
    abnormal = F.normalize(abnormal_text_embeds, dim=-1)
    similarities = torch.einsum("bgd,bkd->bgk", patches, abnormal)
    prompt_count = similarities.shape[-1]
    abnormal_score = prompt_score_temperature * (
        torch.logsumexp(
            similarities / prompt_score_temperature, dim=-1
        )
        - torch.log(similarities.new_tensor(float(prompt_count)))
    )
    normal_score = (patches * normal[:, None, :]).sum(dim=-1)
    return (abnormal_score - normal_score) / temperature


def prompt_diversity_loss(abnormal_text_embeddings):
    """Penalize off-diagonal cosine similarity between abnormal Prompts."""
    prompt_count = abnormal_text_embeddings.shape[-2]
    if prompt_count <= 1:
        return abnormal_text_embeddings.sum() * 0.0
    embeddings = F.normalize(abnormal_text_embeddings, dim=-1)
    gram = embeddings @ embeddings.transpose(-1, -2)
    identity = torch.eye(prompt_count, device=gram.device, dtype=gram.dtype)
    batch_factor = gram.numel() // (prompt_count * prompt_count)
    return ((gram - identity) ** 2).sum() / (
        batch_factor * prompt_count * (prompt_count - 1)
    )


def forward_static_prompt_scores(
    adapter,
    layer_tokens,
    global_embeddings,
    learner,
    clip_model,
    categories,
    temperature=0.07,
    prompt_score_temperature=0.07,
    patch_centers=None,
):
    """Feature-first Static Six Prompt forward shared by train and test."""
    prompts = encode_static_prompts(learner, clip_model, categories)
    # Import locally so the legacy Prompt module does not depend on DDF-3D at
    # import time.  The non-DDF path below is byte-for-byte equivalent in math.
    from models.ddf3d import is_ddf3d_adapter

    ddf3d_output = None
    if is_ddf3d_adapter(adapter):
        feature_output = adapter.forward_features(layer_tokens)
        patch_embeddings = feature_output["patch_embeddings"]
        semantic_logits = static_prompt_anomaly_logits(
            patch_embeddings,
            prompts["normal_text_embed"],
            prompts["abnormal_text_embeds"],
            temperature,
            prompt_score_temperature,
        )
        layer_margins = torch.stack(
            [
                static_prompt_anomaly_logits(
                    layer,
                    prompts["normal_text_embed"],
                    prompts["abnormal_text_embeds"],
                    temperature,
                    prompt_score_temperature,
                )
                for layer in feature_output["projected_layers"]
            ],
            dim=-1,
        )
        ddf3d_output = adapter.enhance_scores(
            semantic_logits,
            layer_margins,
            feature_output,
            patch_centers,
        )
        patch_logits = ddf3d_output["patch_logits"]
    else:
        patch_embeddings = adapter(layer_tokens)
        patch_logits = static_prompt_anomaly_logits(
            patch_embeddings,
            prompts["normal_text_embed"],
            prompts["abnormal_text_embeds"],
            temperature,
            prompt_score_temperature,
        )
    global_logits = static_prompt_anomaly_logits(
        global_embeddings.unsqueeze(1),
        prompts["normal_text_embed"],
        prompts["abnormal_text_embeds"],
        temperature,
        prompt_score_temperature,
    ).squeeze(1)
    output = {
        "patch_logits": patch_logits,
        "global_logits": global_logits,
        "patch_embeddings": patch_embeddings,
        "diversity_embeddings": prompts["abnormal_text_embeds"],
        "normal_text_embed": prompts["normal_text_embed"],
        "abnormal_text_embeds": prompts["abnormal_text_embeds"],
    }
    if ddf3d_output is not None:
        output.update(ddf3d_output)
        output["ddf3d"] = ddf3d_output
    return output


def point_mask_to_patch_targets(point_labels, patch_indices, threshold=0.05):
    """Aggregate point labels into patch anomaly/normal masks."""
    if not 0 <= threshold <= 1:
        raise ValueError("patch anomaly threshold must be in [0, 1]")
    if point_labels.ndim != 2 or patch_indices.ndim != 3:
        raise ValueError("Expected point_labels [B,N] and patch_indices [B,G,M]")
    if point_labels.shape[0] != patch_indices.shape[0]:
        raise ValueError("Batch size mismatch between labels and patch indices")
    valid = ((patch_indices >= 0) & (patch_indices < point_labels.shape[1])).all(
        dim=-1
    )
    safe_indices = patch_indices.long().clamp(0, point_labels.shape[1] - 1)
    gathered = torch.gather(
        point_labels.float(), 1, safe_indices.reshape(point_labels.shape[0], -1)
    ).reshape(patch_indices.shape)
    ratios = gathered.mean(dim=-1)
    anomaly_mask = valid & (ratios >= threshold)
    normal_mask = valid & ~anomaly_mask
    return anomaly_mask, normal_mask, ratios, valid
