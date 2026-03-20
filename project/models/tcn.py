"""Temporal Convolutional Network (TCN) for stock movement prediction."""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """Causal convolution: output at time t only depends on inputs at time <= t."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Residual block with two causal convolutions."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class TCNModel(nn.Module):
    """TCN with exponentially increasing dilation factors."""

    def __init__(self, cfg):
        super().__init__()
        total_input = cfg.input_dim + cfg.context_dim
        self.input_proj = nn.Conv1d(total_input, cfg.hidden_dim, 1)

        blocks = []
        for i in range(cfg.num_layers):
            dilation = 2 ** i
            blocks.append(TCNBlock(cfg.hidden_dim, cfg.tcn_kernel_size, dilation, cfg.dropout))
        self.tcn = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )

    def forward(self, price_seq, context_seq):
        x = torch.cat([price_seq, context_seq], dim=-1)  # (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.input_proj(x)
        x = self.tcn(x)
        x = x[:, :, -1]  # last time step
        return self.head(x)
