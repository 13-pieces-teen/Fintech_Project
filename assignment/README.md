# Week 7 Homework - Trade with Price Action and AI Pattern Recognition

## 文件结构

```
assignment/
├── README.md                          # 本文件
├── answers.md                         # 四道题的完整文字解答
├── Q1_traders_equation.py             # Q1: 交易者方程分析与模拟
├── Q2_candlestick_patterns.py         # Q2: 孤立K线形态 vs 上下文感知分析
├── Q3_cnn_kline_augmentation.py       # Q3: CNN数据增强技术（适当/禁止）
├── Q4_technical_analysis.py           # Q4: 技术分析有效性的多角度验证
└── requirements.txt                   # Python依赖
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 运行代码

```bash
# Question 1: 交易者方程 - 期望值分析、入场时机权衡、蒙特卡洛模拟
python assignment/Q1_traders_equation.py

# Question 2: K线形态分析 - 孤立形态 vs 上下文感知信号对比
python assignment/Q2_candlestick_patterns.py

# Question 3: CNN数据增强 - 增强方法可视化 + 训练对比实验
python assignment/Q3_cnn_kline_augmentation.py

# Question 4: 技术分析有效性 - 自我实现预言、动量、供需、回测
python assignment/Q4_technical_analysis.py
```

## 各题概要

### Q1: Trader's Equation
- 交易者方程: E = P × RR - (1-P)
- 入场时机对胜率P和风险回报比RR产生反向影响
- 量化模型必须联合优化二者以最大化期望值
- 代码: 3D曲面图、盈亏平衡曲线、蒙特卡洛策略对比

### Q2: Candlestick Pattern Flaws
- 孤立K线形态忽略市场上下文(趋势、关键价位、成交量)
- 替代方案: 基于完整市场结构的价格行为分析
- 代码: 孤立射击之星 vs 上下文感知射击之星的胜率对比

### Q3: CNN Data Augmentation for K-lines
- 适当: 价格噪声、缩放、成交量缩放、Cutout、亮度微调
- 禁止: 水平翻转(逆转时间)、垂直翻转(颠倒涨跌)、旋转(破坏坐标轴)
- 代码: 增强效果可视化、有/无增强的CNN训练对比

### Q4: Why Technical Analysis Works
- 自我实现预言、行为金融偏差、信息渐进扩散、供需痕迹
- 代码: 支撑位模拟、动量效应模拟、订单流分析、均线交叉回测
