"""
Question 2: Flaws of Isolated Candlestick Patterns & Context-Based Alternative

传统孤立K线形态（如十字星、射击之星、吞没形态等）的根本性缺陷分析，
以及基于价格行为(Price Action)的上下文分析替代方法。

核心观点：孤立的K线形态忽略了市场上下文(趋势、支撑/阻力、量能等)，
导致信号可靠性极低。替代方法应将价格形态置于完整的市场结构中分析。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_synthetic_ohlcv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成含有趋势结构和均值回归特征的合成OHLCV数据。
    当价格远离均值时有回归倾向,使得阻力位附近的反转形态具有真实的预测力。
    """
    np.random.seed(seed)

    cycle = 8 * np.sin(np.linspace(0, 10 * np.pi, n_bars))
    mean_price = 100 + cycle

    opens, highs, lows, closes, volumes = [], [], [], [], []
    price = mean_price[0]

    for i in range(n_bars):
        reversion = (mean_price[i] - price) * 0.08
        price += reversion + np.random.randn() * 0.8

        o = price + np.random.randn() * 0.3
        c = o + np.random.randn() * 1.2

        deviation = (price - mean_price[i])
        if deviation > 4:
            c = o - abs(np.random.randn()) * 0.8
        elif deviation < -4:
            c = o + abs(np.random.randn()) * 0.8

        h = max(o, c) + abs(np.random.randn()) * 0.7
        l = min(o, c) - abs(np.random.randn()) * 0.7

        v_base = 800 + abs(deviation) * 80
        v = int(np.random.exponential(v_base) + 300)

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        volumes.append(v)
        price = c

    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes
    })


# ============================================================
# 传统孤立K线形态识别（有缺陷的方法）
# ============================================================

def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """识别射击之星(Shooting Star) - 仅基于单根K线形态"""
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    candle_range = df['high'] - df['low']

    return (
        (upper_shadow > 2 * body) &
        (lower_shadow < body * 0.5) &
        (candle_range > candle_range.rolling(20).mean() * 0.5)
    )


def detect_harami(df: pd.DataFrame) -> pd.Series:
    """识别母子线(Harami) - 仅基于两根K线关系"""
    prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
    curr_body = abs(df['close'] - df['open'])
    prev_bearish = df['close'].shift(1) < df['open'].shift(1)
    curr_within = (
        (df[['open', 'close']].max(axis=1) < df[['open', 'close']].shift(1).max(axis=1)) &
        (df[['open', 'close']].min(axis=1) > df[['open', 'close']].shift(1).min(axis=1))
    )

    return prev_bearish & curr_within & (prev_body > curr_body * 1.5)


def detect_engulfing_bullish(df: pd.DataFrame) -> pd.Series:
    """识别看涨吞没(Bullish Engulfing) - 仅基于两根K线关系"""
    prev_bearish = df['close'].shift(1) < df['open'].shift(1)
    curr_bullish = df['close'] > df['open']
    engulfs = (
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    )
    return prev_bearish & curr_bullish & engulfs


# ============================================================
# 基于上下文的价格行为分析（改进方法）
# ============================================================

def compute_market_context(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """计算市场上下文指标"""
    ctx = pd.DataFrame(index=df.index)

    # 趋势方向: 使用SMA斜率
    sma = df['close'].rolling(lookback).mean()
    ctx['trend'] = np.where(sma > sma.shift(5), 1, np.where(sma < sma.shift(5), -1, 0))

    # 支撑/阻力区域: 近期高低点
    ctx['resistance'] = df['high'].rolling(lookback).max()
    ctx['support'] = df['low'].rolling(lookback).min()

    # 价格相对位置 (0=支撑, 1=阻力)
    price_range = ctx['resistance'] - ctx['support']
    ctx['price_position'] = np.where(
        price_range > 0,
        (df['close'] - ctx['support']) / price_range,
        0.5
    )

    # 成交量异常: 当前量 vs 平均量
    avg_vol = df['volume'].rolling(lookback).mean()
    ctx['volume_ratio'] = df['volume'] / avg_vol.replace(0, 1)

    # 波动率: ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    ctx['atr'] = tr.rolling(lookback).mean()

    return ctx


def context_aware_shooting_star(df: pd.DataFrame, ctx: pd.DataFrame) -> pd.Series:
    """
    上下文感知的射击之星信号：
    1. 满足基本射击之星形态
    2. 出现在上升趋势中 (trend > 0)
    3. 价格处于区间上半部 (price_position > 0.65)
    4. 成交量不低于平均水平 (volume_ratio > 0.9)
    """
    basic_pattern = detect_shooting_star(df)
    return (
        basic_pattern &
        (ctx['trend'] == 1) &
        (ctx['price_position'] > 0.65) &
        (ctx['volume_ratio'] > 0.9)
    )


def evaluate_signals(df: pd.DataFrame, signals: pd.Series,
                     forward_bars: int = 10) -> dict:
    """评估信号质量: 信号出现后N根K线的收益"""
    signal_indices = signals[signals].index
    if len(signal_indices) == 0:
        return {'count': 0, 'win_rate': 0, 'avg_return': 0, 'sharpe': 0}

    returns = []
    for idx in signal_indices:
        if idx + forward_bars < len(df):
            entry = df['close'].iloc[idx]
            exit_price = df['close'].iloc[idx + forward_bars]
            ret = (exit_price - entry) / entry
            returns.append(-ret)  # 射击之星是做空信号

    returns = np.array(returns)
    if len(returns) == 0:
        return {'count': 0, 'win_rate': 0, 'avg_return': 0, 'sharpe': 0}

    return {
        'count': len(returns),
        'win_rate': np.mean(returns > 0),
        'avg_return': np.mean(returns),
        'sharpe': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    }


def run_comparison():
    """对比孤立形态 vs 上下文感知形态"""
    print("=" * 60)
    print("对比: 孤立K线形态 vs 上下文感知价格行为分析")
    print("=" * 60)

    results_isolated = []
    results_context = []

    for seed in range(50):
        df = generate_synthetic_ohlcv(2000, seed=seed)
        ctx = compute_market_context(df)

        isolated = detect_shooting_star(df)
        context_sig = context_aware_shooting_star(df, ctx)

        r1 = evaluate_signals(df, isolated)
        r2 = evaluate_signals(df, context_sig)

        results_isolated.append(r1)
        results_context.append(r2)

    avg_isolated = {
        'count': np.mean([r['count'] for r in results_isolated]),
        'win_rate': np.mean([r['win_rate'] for r in results_isolated if r['count'] > 0]),
        'avg_return': np.mean([r['avg_return'] for r in results_isolated if r['count'] > 0]),
    }
    avg_context = {
        'count': np.mean([r['count'] for r in results_context]),
        'win_rate': np.mean([r['win_rate'] for r in results_context if r['count'] > 0]),
        'avg_return': np.mean([r['avg_return'] for r in results_context if r['count'] > 0]),
    }

    print(f"\n孤立射击之星 (Isolated Shooting Star):")
    print(f"  平均信号数: {avg_isolated['count']:.1f}")
    print(f"  平均胜率:   {avg_isolated['win_rate']:.2%}")
    print(f"  平均收益:   {avg_isolated['avg_return']:.4%}")

    print(f"\n上下文感知射击之星 (Context-Aware Shooting Star):")
    print(f"  平均信号数: {avg_context['count']:.1f}")
    print(f"  平均胜率:   {avg_context['win_rate']:.2%}")
    print(f"  平均收益:   {avg_context['avg_return']:.4%}")

    return results_isolated, results_context


def plot_pattern_in_context(seed: int = 5):
    """可视化: 在K线图上标注孤立 vs 上下文信号"""
    df = generate_synthetic_ohlcv(200, seed=seed)
    ctx = compute_market_context(df)

    isolated = detect_shooting_star(df)
    context_sig = context_aware_shooting_star(df, ctx)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1]})

    # K线图
    ax = axes[0]
    colors = ['green' if c >= o else 'red' for o, c in zip(df['open'], df['close'])]
    for i in range(len(df)):
        ax.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]], color='gray', linewidth=0.5)
        ax.plot([i, i], [df['open'].iloc[i], df['close'].iloc[i]], color=colors[i], linewidth=2)

    ax.plot(df.index, ctx['resistance'], 'r--', alpha=0.5, linewidth=1, label='Resistance')
    ax.plot(df.index, ctx['support'], 'g--', alpha=0.5, linewidth=1, label='Support')

    isolated_only = isolated & ~context_sig
    for idx in df.index[isolated_only]:
        ax.scatter(idx, df['high'].iloc[idx] + 1, marker='v', color='orange',
                   s=80, zorder=5)
    for idx in df.index[context_sig]:
        ax.scatter(idx, df['high'].iloc[idx] + 1, marker='v', color='red',
                   s=120, zorder=5, edgecolors='black')

    ax.legend(handles=[
        mpatches.Patch(color='orange', label='Isolated Pattern (Low Reliability)'),
        mpatches.Patch(color='red', label='Context-Aware Signal (High Reliability)'),
    ], fontsize=9)
    ax.set_title('Candlestick Patterns: Isolated vs Context-Aware Detection')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.2)

    # 成交量
    axes[1].bar(df.index, df['volume'], color=colors, alpha=0.6)
    axes[1].set_ylabel('Volume')
    axes[1].set_xlabel('Bar Index')
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('assignment/Q2_context_vs_isolated.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    run_comparison()
    plot_pattern_in_context()
