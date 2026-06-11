"""Model components for structural community representation using Mamba."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SubgraphTokenEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        return self.net(x)


class MambaSequenceEncoder(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, dropout: float = 0.1):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        out = self.mamba(x)
        out = self.dropout(out)
        return out


class MambaCommunityModel(nn.Module):
    def __init__(
        self,
        token_dim: int,
        token_hidden_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_encoder = SubgraphTokenEncoder(token_dim, token_hidden_dim, d_model, dropout)
        self.sequence_encoder = MambaSequenceEncoder(d_model, d_state, d_conv, dropout)
        self.norm = nn.LayerNorm(d_model)

    def encode_walks(self, walk_tokens: torch.Tensor) -> torch.Tensor:
        # walk_tokens: (B, L, F)
        h = self.token_encoder(walk_tokens)
        h = self.sequence_encoder(h)
        pooled = h.mean(dim=1)
        return self.norm(pooled)

    def forward(self, tokens_by_node: torch.Tensor) -> torch.Tensor:
        # tokens_by_node: (N, num_walks, L, F)
        N, W, L, F_dim = tokens_by_node.shape
        flat = tokens_by_node.view(N * W, L, F_dim)
        walk_emb = self.encode_walks(flat)  # (N*W, d_model)
        walk_emb = walk_emb.view(N, W, -1)
        node_emb = walk_emb.mean(dim=1)
        node_emb = F.normalize(node_emb, p=2, dim=-1)
        return node_emb
 