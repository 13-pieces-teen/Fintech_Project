"""
Market Structure Feature Engineering.

Extracts contextual features that encode the market's structural state,
directly inspired by the homework Q2 insight: context-aware analysis
significantly outperforms isolated pattern recognition.
"""

import numpy as np
import pandas as pd


def compute_market_structure_features(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Compute market structure context features from OHLCV data.

    Returns DataFrame with 8 features:
        1. trend_strength   - Normalized SMA slope (trend direction & strength)
        2. sr_distance      - Distance to nearest support/resistance (normalized by ATR)
        3. price_position   - Price position within recent range [0, 1]
        4. volume_ratio     - Current volume / avg volume
        5. volatility_regime - ATR / price (normalized volatility)
        6. momentum         - Rate of change over lookback period
        7. rsi              - Relative Strength Index
        8. volume_trend     - Volume SMA slope (rising/falling activity)
    """
    ctx = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)

    sma = close.rolling(lookback).mean()
    sma_slope = (sma - sma.shift(5)) / (sma.shift(5) + 1e-8)
    ctx["trend_strength"] = sma_slope

    resistance = high.rolling(lookback).max()
    support = low.rolling(lookback).min()
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(lookback).mean()

    dist_to_resistance = resistance - close
    dist_to_support = close - support
    nearest_sr = pd.concat([dist_to_resistance, dist_to_support], axis=1).min(axis=1)
    ctx["sr_distance"] = nearest_sr / (atr + 1e-8)

    price_range = resistance - support
    ctx["price_position"] = np.where(
        price_range > 0,
        (close - support) / price_range,
        0.5,
    )

    avg_vol = volume.rolling(lookback).mean()
    ctx["volume_ratio"] = volume / (avg_vol + 1e-8)

    ctx["volatility_regime"] = atr / (close + 1e-8)

    ctx["momentum"] = close.pct_change(lookback)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    ctx["rsi"] = 100 - 100 / (1 + rs)
    ctx["rsi"] = ctx["rsi"] / 100.0  # normalize to [0, 1]

    vol_sma = volume.rolling(lookback).mean()
    ctx["volume_trend"] = (vol_sma - vol_sma.shift(5)) / (vol_sma.shift(5) + 1e-8)

    return ctx


def normalize_ohlcv_window(window: np.ndarray) -> np.ndarray:
    """
    Normalize a single OHLCV window using the first close price,
    making the model invariant to absolute price levels.

    Args:
        window: shape (seq_len, 5) — columns are [O, H, L, C, V]
    Returns:
        Normalized window of same shape.
    """
    out = window.copy()
    ref_price = window[0, 3]  # first close price
    if ref_price > 0:
        out[:, :4] = out[:, :4] / ref_price - 1.0  # price cols → relative returns
    ref_vol = window[:, 4].mean()
    if ref_vol > 0:
        out[:, 4] = out[:, 4] / ref_vol  # volume → ratio to window mean
    return out
