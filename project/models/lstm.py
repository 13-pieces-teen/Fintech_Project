"""LSTM and GRU baselines for stock movement prediction."""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """2-layer LSTM with concatenated price + context features."""

    def __init__(self, cfg):
        super().__init__()
        total_input = cfg.input_dim + cfg.context_dim
        self.rnn = nn.LSTM(
            input_size=total_input,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            batch_first=True,
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
            price_seq:   (B, T, 5)
            context_seq: (B, T, 8)
        Returns:
            logits: (B, 1)
        """
        x = torch.cat([price_seq, context_seq], dim=-1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class GRUModel(nn.Module):
    """2-layer GRU baseline."""

    def __init__(self, cfg):
        super().__init__()
        total_input = cfg.input_dim + cfg.context_dim
        self.rnn = nn.GRU(
            input_size=total_input,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, price_seq, context_seq):
        x = torch.cat([price_seq, context_seq], dim=-1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])
