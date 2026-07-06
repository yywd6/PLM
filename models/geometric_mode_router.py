"""K-specific geometric routing and prompt-token residual modulation."""

import torch
import torch.nn as nn


class GeometricModeRouter(nn.Module):
    """Route invariant graph nodes into K latent geometric anomaly modes."""

    def __init__(self, graph_dim=128, num_modes=10, temperature=0.2):
        super().__init__()
        if temperature <= 0:
            raise ValueError("mode router temperature must be positive")
        self.num_modes = int(num_modes)
        self.temperature = float(temperature)
        self.node_router = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
            nn.Linear(graph_dim, self.num_modes),
        )
        # Low temperature amplifies random initial logits, so initialize the
        # final router near zero and start from an almost uniform assignment.
        nn.init.normal_(self.node_router[-1].weight, std=0.01)
        nn.init.zeros_(self.node_router[-1].bias)
        self.abnormal_gate = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.GELU(),
            nn.Linear(graph_dim, 1),
        )
        nn.init.normal_(self.abnormal_gate[-1].weight, std=0.01)
        nn.init.constant_(self.abnormal_gate[-1].bias, -2.0)

    def forward(self, graph_features):
        node_mode_logits = self.node_router(graph_features)  # [B,G,K]
        abnormal_gate_logits = self.abnormal_gate(graph_features).squeeze(-1)
        mode_logits = node_mode_logits.mean(dim=1)
        node_mode_weights = torch.softmax(
            node_mode_logits / self.temperature, dim=-1
        )  # [B,G,K]
        # A sample mode weight is the fraction of its graph patches assigned
        # to that mode. This preserves patch-level geometric variation instead
        # of applying softmax after graph-wide logit averaging.
        mode_weights = node_mode_weights.mean(dim=1)
        patch_attention = node_mode_weights.transpose(1, 2)
        patch_attention = patch_attention / patch_attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)  # [B,K,G]
        mode_features = torch.einsum("bkg,bgc->bkc", patch_attention, graph_features)
        return {
            "node_mode_logits": node_mode_logits,
            "abnormal_gate_logits": abnormal_gate_logits,
            "mode_logits": mode_logits,
            "node_mode_weights": node_mode_weights,
            "mode_weights": mode_weights,
            "patch_mode_attention": patch_attention,
            "mode_features": mode_features,
        }


class ModeSpecificPromptModulator(nn.Module):
    """Generate one residual tensor for every sample, mode, and abnormal token."""

    def __init__(
        self,
        graph_dim=128,
        num_modes=10,
        num_abnormal_tokens=4,
        token_dim=1664,
        hidden_dim=512,
        residual_scale=0.1,
    ):
        super().__init__()
        self.num_modes = int(num_modes)
        self.num_abnormal_tokens = int(num_abnormal_tokens)
        self.token_dim = int(token_dim)
        self.residual_scale = float(residual_scale)
        self.mode_embeddings = nn.Parameter(torch.empty(self.num_modes, graph_dim))
        nn.init.normal_(self.mode_embeddings, std=0.02)
        self.geometry_projection = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.mode_gate = nn.Linear(graph_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(
            hidden_dim,
            self.num_abnormal_tokens * self.token_dim,
            bias=False,
        )
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, mode_features):
        batch_size, mode_count, _ = mode_features.shape
        if mode_count != self.num_modes:
            raise ValueError(f"Expected {self.num_modes} modes, got {mode_count}")
        mode_identity = self.mode_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        geometry_hidden = self.geometry_projection(mode_features)
        gate = 2.0 * torch.sigmoid(self.mode_gate(mode_identity))
        # Mode identity can only gate geometry; it cannot generate a residual
        # through an independent bias or concatenation shortcut.
        residual = torch.tanh(self.output_projection(geometry_hidden * gate))
        return self.residual_scale * residual.reshape(
            batch_size, self.num_modes, self.num_abnormal_tokens, self.token_dim
        )
