"""
PatchTST — Patch-based Time Series Transformer.

Reference: Nie et al., "A Time Series is Worth 64 Words:
Long-term Forecasting with Transformers." ICLR 2023.
"""

import math
import torch
import torch.nn as nn


class PatchTSTModel(nn.Module):
    """
    Simplified PatchTST for stock movement prediction.
    Splits the input sequence into non-overlapping (or strided) patches,
    projects each patch into an embedding, then applies a Transformer encoder.
    """

    def __init__(self, cfg):
        super().__init__()
        total_input = cfg.input_dim + cfg.context_dim
        self.patch_len = cfg.patch_len
        self.stride = cfg.stride

        self.patch_proj = nn.Linear(total_input * cfg.patch_len, cfg.hidden_dim)

        max_patches = 200
        pe = torch.zeros(max_patches, cfg.hidden_dim)
        position = torch.arange(0, max_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, cfg.hidden_dim, 2).float()
            * (-math.log(10000.0) / cfg.hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, price_seq, context_seq):
        x = torch.cat([price_seq, context_seq], dim=-1)  # (B, T, C)
        B, T, C = x.shape

        patches = []
        for start in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, start:start + self.patch_len, :]  # (B, patch_len, C)
            patches.append(patch.reshape(B, -1))           # (B, patch_len * C)
        patches = torch.stack(patches, dim=1)               # (B, num_patches, patch_len * C)

        x = self.patch_proj(patches)                        # (B, num_patches, hidden_dim)
        x = x + self.pe[:, :x.size(1)]
        x = self.encoder(x)
        return self.head(x[:, -1, :])
