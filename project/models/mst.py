"""
Market Structure Transformer (MST) — the core contribution of this project.

Unlike baselines that simply concatenate price and context features, MST uses
a Cross-Attention mechanism to let the price representation dynamically attend
to market structure context. This mirrors the human trader's process: first
read the price action, then interpret it through the lens of market context
(trend, support/resistance, volume profile, volatility regime).

Architecture:
    1. Separate embedding of price sequence and context sequence
    2. Cross-Attention: price tokens (query) attend to context tokens (key/value)
    3. Self-Attention Transformer encoder on the fused representation
    4. Classification/Regression head on the [CLS]-like final token
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention: query comes from price, key/value come from context.
    This lets each price time-step selectively attend to relevant market
    structure information.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, price_emb, context_emb):
        """
        Args:
            price_emb:   (B, T, D) — query
            context_emb: (B, T, D) — key & value
        Returns:
            fused: (B, T, D)
        """
        attn_out, _ = self.cross_attn(
            query=price_emb, key=context_emb, value=context_emb,
        )
        x = self.norm1(price_emb + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class MarketStructureTransformer(nn.Module):
    """
    MST: Market Structure Transformer.

    Separate embeddings → Cross-Attention fusion → Self-Attention encoding → Head
    """

    def __init__(self, cfg):
        super().__init__()

        self.price_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.price_pos = PositionalEncoding(cfg.hidden_dim)
        self.context_pos = PositionalEncoding(cfg.hidden_dim)

        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_cross_attn_layers)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.self_attn_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, price_seq, context_seq):
        """
        Args:
            price_seq:   (B, T, 5)  — normalized OHLCV
            context_seq: (B, T, 8)  — market structure features
        Returns:
            logits: (B, 1)
        """
        price_emb = self.price_pos(self.price_proj(price_seq))
        context_emb = self.context_pos(self.context_proj(context_seq))

        fused = price_emb
        for cross_layer in self.cross_attn_layers:
            fused = cross_layer(fused, context_emb)

        encoded = self.self_attn_encoder(fused)

        return self.head(encoded[:, -1, :])
