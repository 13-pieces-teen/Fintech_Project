"""Centralized configuration for all experiments."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    market: str = "us"
    tickers_us: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "V", "JNJ",
        "WMT", "PG", "MA", "UNH", "HD",
        "DIS", "BAC", "XOM", "PFE", "CSCO",
    ])
    tickers_cn: List[str] = field(default_factory=lambda: [
        "600519", "601318", "600036", "000858", "600276",
        "601166", "000333", "002415", "600900", "601888",
    ])
    start_date: str = "2015-01-01"
    end_date: str = "2025-12-31"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seq_len: int = 60
    pred_len: int = 1
    label_type: str = "classification"  # "classification" or "regression"
    data_dir: str = "project/data/raw"


@dataclass
class ModelConfig:
    name: str = "mst"
    # Shared
    input_dim: int = 5       # OHLCV
    context_dim: int = 8     # market structure features
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    num_heads: int = 4
    # TCN specific
    tcn_kernel_size: int = 3
    # PatchTST specific
    patch_len: int = 12
    stride: int = 6
    # MST specific
    num_cross_attn_layers: int = 2


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    patience: int = 15
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "project/checkpoints"
    log_dir: str = "project/logs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
