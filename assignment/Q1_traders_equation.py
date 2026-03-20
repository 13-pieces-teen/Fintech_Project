"""
Question 1: Trader's Equation - Entry Timing, Risk-Reward Ratio, and Probability

交易者方程中入场时机、风险回报比(RR)和概率之间的关系分析，
以及这种关系对量化模型设计的重要性。

核心公式: Expectancy = P(win) × RR - P(loss) = P × RR - (1 - P)
其中 P = 胜率, RR = 风险回报比(盈亏比)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_expectancy(win_rate: float, risk_reward: float) -> float:
    """
    E = P × RR - (1 - P)
    """
    return win_rate * risk_reward - (1 - win_rate)


def simulate_entry_timing_tradeoff(n_scenarios: int = 100):
    """
    模拟入场时机对胜率和RR的反向影响：
    - 越早入场 → RR越高,但胜率越低(确认信号少)
    - 越晚入场(更多确认) → 胜率越高,但RR越低(入场价格不利)
    """
    aggressiveness = np.linspace(0, 1, n_scenarios)

    # 激进入场: 高RR, 低胜率; 保守入场: 低RR, 高胜率
    rr_values = 0.5 + 4.0 * aggressiveness
    win_rates = 0.70 - 0.40 * aggressiveness

    expectancies = np.array([
        calculate_expectancy(p, rr)
        for p, rr in zip(win_rates, rr_values)
    ])

    return aggressiveness, win_rates, rr_values, expectancies


def plot_expectancy_surface():
    """绘制胜率-RR-期望收益的3D曲面图"""
    win_rates = np.linspace(0.1, 0.9, 50)
    rr_values = np.linspace(0.5, 5.0, 50)
    W, R = np.meshgrid(win_rates, rr_values)
    E = W * R - (1 - W)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(W, R, E, cmap='RdYlGn', alpha=0.8)
    ax1.set_xlabel('Win Rate (P)')
    ax1.set_ylabel('Risk-Reward Ratio (RR)')
    ax1.set_zlabel('Expectancy (E)')
    ax1.set_title("Trader's Equation: E = P × RR - (1 - P)")

    zero_contour = ax1.contour(W, R, E, levels=[0], colors='black', linewidths=2)
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Breakeven curve: P × RR = 1 - P → P = 1 / (1 + RR)
    ax2 = fig.add_subplot(122)
    rr_line = np.linspace(0.5, 5.0, 100)
    breakeven_wr = 1.0 / (1.0 + rr_line)

    ax2.plot(rr_line, breakeven_wr, 'k-', linewidth=2, label='Breakeven: E = 0')
    ax2.fill_between(rr_line, breakeven_wr, 1.0, alpha=0.3, color='green', label='Profitable Zone (E > 0)')
    ax2.fill_between(rr_line, 0, breakeven_wr, alpha=0.3, color='red', label='Loss Zone (E < 0)')

    ax2.scatter([2.0], [0.55], color='blue', s=100, zorder=5, label='Balanced Strategy')
    ax2.scatter([4.0], [0.30], color='orange', s=100, zorder=5, label='Aggressive Entry (High RR, Low P)')
    ax2.scatter([1.0], [0.65], color='purple', s=100, zorder=5, label='Conservative Entry (Low RR, High P)')

    ax2.set_xlabel('Risk-Reward Ratio (RR)')
    ax2.set_ylabel('Win Rate (P)')
    ax2.set_title('Breakeven Curve & Strategy Zones')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.5, 5.0)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assignment/Q1_expectancy_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_entry_timing_tradeoff():
    """绘制入场时机权衡图"""
    aggressiveness, win_rates, rr_values, expectancies = simulate_entry_timing_tradeoff()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(aggressiveness, win_rates, 'b-', linewidth=2, label='Win Rate')
    axes[0].plot(aggressiveness, rr_values / 5, 'r-', linewidth=2, label='RR (scaled /5)')
    axes[0].set_xlabel('Entry Aggressiveness (0=Conservative, 1=Aggressive)')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Win Rate vs RR Tradeoff')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    colors = ['green' if e > 0 else 'red' for e in expectancies]
    axes[1].bar(aggressiveness, expectancies, width=0.012, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Entry Aggressiveness')
    axes[1].set_ylabel('Expectancy')
    axes[1].set_title('Expectancy vs Entry Timing')
    axes[1].grid(True, alpha=0.3)

    optimal_idx = np.argmax(expectancies)
    axes[2].plot(aggressiveness, expectancies, 'g-', linewidth=2)
    axes[2].scatter(
        aggressiveness[optimal_idx], expectancies[optimal_idx],
        color='red', s=150, zorder=5,
        label=f'Optimal: aggr={aggressiveness[optimal_idx]:.2f}, '
              f'E={expectancies[optimal_idx]:.3f}'
    )
    axes[2].set_xlabel('Entry Aggressiveness')
    axes[2].set_ylabel('Expectancy')
    axes[2].set_title('Finding the Optimal Entry Timing')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assignment/Q1_entry_timing_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.show()


def monte_carlo_strategy_comparison(n_trades: int = 1000, n_simulations: int = 500):
    """
    蒙特卡洛模拟对比三种策略：
    1. 激进入场 (高RR=3.5, 低胜率=0.30)
    2. 保守入场 (低RR=1.2, 高胜率=0.62)
    3. 平衡入场 (中RR=2.0, 中胜率=0.50)
    """
    strategies = {
        'Aggressive (RR=3.5, P=0.30)': (0.30, 3.5),
        'Conservative (RR=1.2, P=0.62)': (0.62, 1.2),
        'Balanced (RR=2.0, P=0.50)': (0.50, 2.0),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    risk_per_trade = 100  # 每笔固定风险金额

    for name, (win_rate, rr) in strategies.items():
        expectancy = calculate_expectancy(win_rate, rr)
        final_pnls = []

        for _ in range(n_simulations):
            wins = np.random.random(n_trades) < win_rate
            pnl_per_trade = np.where(wins, risk_per_trade * rr, -risk_per_trade)
            cumulative_pnl = np.cumsum(pnl_per_trade)
            final_pnls.append(cumulative_pnl[-1])

        final_pnls = np.array(final_pnls)
        ax.hist(final_pnls, bins=50, alpha=0.5,
                label=f'{name}\nE={expectancy:.2f}, '
                      f'Mean={np.mean(final_pnls):.0f}, '
                      f'Std={np.std(final_pnls):.0f}')

    ax.set_xlabel('Final P&L after 1000 trades')
    ax.set_ylabel('Frequency')
    ax.set_title('Monte Carlo: Strategy Comparison (500 simulations × 1000 trades)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assignment/Q1_monte_carlo_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("=" * 70)
    print("Question 1: Trader's Equation Analysis")
    print("=" * 70)

    print("\n--- 交易者方程期望值示例 ---")
    examples = [
        (0.30, 3.5, "激进入场"),
        (0.50, 2.0, "平衡入场"),
        (0.62, 1.2, "保守入场"),
        (0.40, 1.0, "差策略"),
    ]
    for wr, rr, desc in examples:
        e = calculate_expectancy(wr, rr)
        print(f"  {desc}: P={wr:.2f}, RR={rr:.1f} → E={e:.3f} "
              f"({'Profitable' if e > 0 else 'Losing'})")

    print("\n--- 生成图表 ---")
    plot_expectancy_surface()
    plot_entry_timing_tradeoff()
    monte_carlo_strategy_comparison()
    print("Done!")
