"""PyTorch Dataset and DataLoader construction for stock movement prediction."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .features import compute_market_structure_features, normalize_ohlcv_window


class StockDataset(Dataset):
    """
    Sliding-window dataset for stock movement prediction.

    Each sample contains:
        - price_seq:   (seq_len, 5)  normalized OHLCV
        - context_seq: (seq_len, 8)  market structure features
        - label:       scalar — 1/0 for classification, float for regression
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 60,
        pred_len: int = 1,
        label_type: str = "classification",
        lookback: int = 20,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_type = label_type

        ohlcv = df[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
        ctx_df = compute_market_structure_features(df, lookback=lookback)
        ctx = ctx_df.values.astype(np.float32)

        valid_start = max(lookback + 14, seq_len)
        self.ohlcv = ohlcv[valid_start:]
        self.ctx = ctx[valid_start:]
        self.closes = df["close"].values[valid_start:].astype(np.float32)

        self.n_samples = len(self.ohlcv) - seq_len - pred_len

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        s, e = idx, idx + self.seq_len
        price_window = normalize_ohlcv_window(self.ohlcv[s:e])
        context_window = self.ctx[s:e]

        nan_mask = np.isnan(context_window)
        context_window = np.where(nan_mask, 0.0, context_window)

        future_close = self.closes[e + self.pred_len - 1]
        current_close = self.closes[e - 1]
        ret = (future_close - current_close) / (current_close + 1e-8)

        if self.label_type == "classification":
            label = np.float32(1.0 if ret > 0 else 0.0)
        else:
            label = np.float32(ret)

        return (
            torch.from_numpy(price_window),
            torch.from_numpy(context_window),
            torch.tensor(label),
        )


def create_dataloaders(
    all_data: dict[str, pd.DataFrame],
    seq_len: int = 60,
    pred_len: int = 1,
    label_type: str = "classification",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from a dict of ticker → DataFrame.
    Split is done temporally (no future leakage).
    """
    train_datasets, val_datasets, test_datasets = [], [], []

    for ticker, df in all_data.items():
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        for split_df, container in [
            (df.iloc[:train_end], train_datasets),
            (df.iloc[train_end:val_end], val_datasets),
            (df.iloc[val_end:], test_datasets),
        ]:
            if len(split_df) > seq_len + pred_len + 40:
                ds = StockDataset(split_df, seq_len, pred_len, label_type)
                if len(ds) > 0:
                    container.append(ds)

    train_ds = torch.utils.data.ConcatDataset(train_datasets) if train_datasets else []
    val_ds = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else []
    test_ds = torch.utils.data.ConcatDataset(test_datasets) if test_datasets else []

    def make_loader(ds, shuffle):
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        make_loader(train_ds, shuffle=True),
        make_loader(val_ds, shuffle=False),
        make_loader(test_ds, shuffle=False),
    )
