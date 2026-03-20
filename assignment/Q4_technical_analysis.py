"""
Question 4: Why Should Technical Analysis Work?

技术分析为何有效的理论分析与实证验证。

核心论点:
1. 自我实现的预言 (Self-Fulfilling Prophecy)
2. 市场心理学与行为偏差 (Behavioral Finance)
3. 供需失衡在价格中留下痕迹 (Supply-Demand Footprints)
4. 信息的渐进扩散 (Gradual Information Diffusion)
5. 机构资金的有序流动 (Institutional Order Flow)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================
# 论点1: 自我实现预言的模拟
# ============================================================

def simulate_self_fulfilling_prophecy(n_traders: int = 1000, n_steps: int = 200):
    """
    模拟: 当足够多的交易者关注同一技术水平(如支撑位)时,
    集体行为会让该水平真正成为有效的支撑/阻力。
    """
    support_level = 100.0
    ta_ratio_list = [0.0, 0.3, 0.6, 0.9]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, ta_ratio in zip(axes.flatten(), ta_ratio_list):
        n_ta_traders = int(n_traders * ta_ratio)
        n_random_traders = n_traders - n_ta_traders

        prices = [105.0]

        for _ in range(n_steps):
            current_price = prices[-1]

            random_orders = np.random.randn(n_random_traders) * 0.5

            if n_ta_traders > 0:
                distance_to_support = (current_price - support_level) / support_level
                if distance_to_support < 0.03:
                    ta_orders = np.abs(np.random.randn(n_ta_traders)) * 0.3
                elif distance_to_support < 0:
                    ta_orders = np.abs(np.random.randn(n_ta_traders)) * 0.8
                else:
                    ta_orders = np.random.randn(n_ta_traders) * 0.1
            else:
                ta_orders = np.array([])

            all_orders = np.concatenate([random_orders, ta_orders])
            price_impact = np.mean(all_orders) * 0.5
            new_price = current_price + price_impact + np.random.randn() * 0.3
            prices.append(new_price)

        ax.plot(prices, 'b-', linewidth=0.8)
        ax.axhline(y=support_level, color='red', linestyle='--', linewidth=1.5,
                    label=f'Support @ {support_level}')
        ax.set_title(f'TA Traders: {ta_ratio:.0%} ({n_ta_traders}/{n_traders})')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        touches = sum(1 for p in prices if abs(p - support_level) < 1.5)
        bounces = sum(1 for i in range(1, len(prices))
                      if prices[i - 1] < support_level + 1.5 and prices[i] > prices[i - 1])
        ax.text(0.02, 0.02, f'Touches: {touches}, Bounces: {bounces}',
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Self-Fulfilling Prophecy: More TA Traders → Stronger Support Level',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('assignment/Q4_self_fulfilling.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 论点2: 行为金融与趋势动量
# ============================================================

def simulate_behavioral_momentum(n_steps: int = 500, seed: int = 42):
    """
    模拟: 信息渐进扩散导致趋势延续(动量效应)
    - 信息不是瞬间被所有人获知的
    - 先知先觉者先买入 → 价格开始上涨
    - 后知后觉者看到涨势跟入 → 趋势延续
    - 最终过度反应 → 反转
    """
    np.random.seed(seed)

    fundamental_value = np.ones(n_steps) * 100
    fundamental_value[100:] = 120  # t=100时基本面突变

    # 3类交易者: 知情者、趋势追随者、噪声交易者
    price = 100.0
    prices = [price]

    informed_pct = 0.0
    follower_pct = 0.0

    informed_history = [0]
    follower_history = [0]

    for t in range(1, n_steps):
        # 知情者逐渐获知信息 (S-curve)
        if t > 100:
            informed_pct = min(1.0, informed_pct + 0.02)

        # 趋势追随者根据近期趋势决策
        if t > 110:
            recent_return = (prices[-1] - prices[-10]) / prices[-10] if len(prices) > 10 else 0
            follower_pct = np.clip(recent_return * 10, -1, 1)

        informed_demand = informed_pct * (fundamental_value[t] - price) * 0.05
        follower_demand = follower_pct * 2.0
        noise = np.random.randn() * 0.5

        price_change = informed_demand + follower_demand + noise
        price = max(price + price_change, 50)
        prices.append(price)
        informed_history.append(informed_pct)
        follower_history.append(max(0, follower_pct))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(prices, 'b-', linewidth=1, label='Market Price')
    axes[0].plot(fundamental_value, 'r--', linewidth=1.5, label='Fundamental Value')
    axes[0].axvline(x=100, color='gray', linestyle=':', label='Information Event')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Gradual Information Diffusion → Trend/Momentum')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(range(n_steps), informed_history, alpha=0.5, color='green',
                         label='Informed Traders %')
    axes[1].fill_between(range(n_steps), follower_history, alpha=0.5, color='orange',
                         label='Trend Followers %')
    axes[1].set_ylabel('Participation')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    returns = np.diff(prices) / prices[:-1]
    momentum = pd.Series(returns).rolling(20).mean().values
    axes[2].plot(momentum, 'purple', linewidth=1, label='20-bar Momentum')
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Momentum')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assignment/Q4_behavioral_momentum.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 论点3: 支撑/阻力的供需解释
# ============================================================

def simulate_support_resistance_orderflow(n_steps: int = 300, seed: int = 42):
    """
    模拟: 支撑/阻力位对应历史成交密集区,
    这些区域存在大量未执行的限价单(供给/需求)
    """
    np.random.seed(seed)

    price = 100.0
    prices = [price]
    order_book = defaultdict(float)

    for t in range(n_steps):
        # 在当前价格附近随机放置限价单
        for _ in range(10):
            order_price = round(price + np.random.randn() * 2, 0)
            order_size = np.random.exponential(50)
            if order_price > price:
                order_book[order_price] -= order_size  # 卖单(阻力)
            else:
                order_book[order_price] += order_size  # 买单(支撑)

        nearby_orders = sum(order_book.get(round(price + dp, 0), 0) for dp in np.arange(-1, 2))
        order_pressure = nearby_orders * 0.001
        price += np.random.randn() * 0.8 + order_pressure
        prices.append(price)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(prices, 'b-', linewidth=0.8)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Price Movement with Order Flow Dynamics')
    axes[0].grid(True, alpha=0.3)

    price_levels = sorted(order_book.keys())
    net_orders = [order_book[p] for p in price_levels]

    colors = ['green' if v > 0 else 'red' for v in net_orders]
    axes[1].barh(price_levels, net_orders, color=colors, alpha=0.6, height=0.8)
    axes[1].set_xlabel('Net Order Size (+ = Buy/Support, - = Sell/Resistance)')
    axes[1].set_ylabel('Price Level')
    axes[1].set_title('Order Book: Supply & Demand Zones')
    axes[1].axvline(x=0, color='black', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    y_min, y_max = min(prices) - 5, max(prices) + 5
    axes[1].set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig('assignment/Q4_supply_demand.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 论点4: 简单移动平均交叉策略回测
# ============================================================

def backtest_ma_crossover(n_steps: int = 3000, seed: int = 42):
    """
    回测简单双均线交叉策略,验证趋势跟踪的有效性。
    使用 regime-switching 模型生成具有持续趋势特征的价格序列。
    """
    np.random.seed(seed)

    prices = [100.0]
    regime = 1
    for t in range(1, n_steps):
        if np.random.random() < 0.003:
            regime *= -1
        drift = regime * 0.15
        prices.append(prices[-1] * (1 + drift / 100 + np.random.randn() * 0.012))

    prices = np.array(prices)
    df = pd.DataFrame({'close': prices})

    df['sma_fast'] = df['close'].rolling(10).mean()
    df['sma_slow'] = df['close'].rolling(50).mean()

    df['signal'] = 0
    df.loc[df['sma_fast'] > df['sma_slow'], 'signal'] = 1
    df.loc[df['sma_fast'] < df['sma_slow'], 'signal'] = -1

    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    df['buy_hold_cumret'] = (1 + df['returns']).cumprod()
    df['strategy_cumret'] = (1 + df['strategy_returns']).cumprod()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df['close'], 'gray', linewidth=0.8, label='Price')
    axes[0].plot(df['sma_fast'], 'blue', linewidth=1, label='SMA(10)')
    axes[0].plot(df['sma_slow'], 'red', linewidth=1, label='SMA(50)')
    axes[0].set_ylabel('Price')
    axes[0].set_title('MA Crossover Strategy Backtest')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    colors = df['signal'].map({1: 'green', -1: 'red', 0: 'gray'})
    axes[1].bar(df.index, df['signal'], color=colors, alpha=0.5, width=1)
    axes[1].set_ylabel('Position')
    axes[1].set_title('Trading Signal (+1 = Long, -1 = Short)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df['buy_hold_cumret'], 'gray', linewidth=1, label='Buy & Hold')
    axes[2].plot(df['strategy_cumret'], 'blue', linewidth=1.5, label='MA Crossover')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Cumulative Return')
    axes[2].set_title('Strategy Performance vs Buy & Hold')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assignment/Q4_ma_backtest.png', dpi=150, bbox_inches='tight')
    plt.show()

    valid = df.dropna()
    strategy_sharpe = (valid['strategy_returns'].mean() / valid['strategy_returns'].std()
                       * np.sqrt(252))
    bh_sharpe = valid['returns'].mean() / valid['returns'].std() * np.sqrt(252)

    print(f"\n--- MA Crossover Backtest Results ---")
    print(f"  策略夏普比率: {strategy_sharpe:.3f}")
    print(f"  买入持有夏普: {bh_sharpe:.3f}")
    print(f"  策略累计收益: {valid['strategy_cumret'].iloc[-1] - 1:.2%}")
    print(f"  买入持有累计: {valid['buy_hold_cumret'].iloc[-1] - 1:.2%}")


if __name__ == '__main__':
    print("=" * 60)
    print("Question 4: Why Should Technical Analysis Work?")
    print("=" * 60)

    print("\n--- 1. Self-Fulfilling Prophecy Simulation ---")
    simulate_self_fulfilling_prophecy()

    print("\n--- 2. Behavioral Momentum (Information Diffusion) ---")
    simulate_behavioral_momentum()

    print("\n--- 3. Supply-Demand / Order Flow ---")
    simulate_support_resistance_orderflow()

    print("\n--- 4. MA Crossover Backtest ---")
    backtest_ma_crossover()
