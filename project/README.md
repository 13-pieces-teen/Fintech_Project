# Context Matters: Market Structure-Aware Transformer for Stock Movement Prediction

## Project Overview

本项目提出 **Market Structure Transformer (MST)**，一种融合市场结构上下文信息的深度学习模型，用于股票价格走势预测。核心创新在于：传统方法仅使用原始价格序列作为输入，而 MST 显式建模市场结构特征（趋势状态、支撑/阻力位、成交量分布、波动率机制），通过 Cross-Attention 机制将上下文信息注入价格序列的表征学习中。

**项目类别**: Original Project (Category 1)

### Motivation

在课程作业 Q2 中，我们分析了孤立 K 线形态识别的根本缺陷——缺乏市场上下文（趋势、关键价位、量能）导致信号可靠性极低。上下文感知的分析方法显著优于孤立形态识别。本项目将这一洞察推广到深度学习框架中：**让模型不仅"看到"价格序列，还"理解"市场结构**。

### Research Questions

1. 显式编码市场结构特征是否能显著提升深度学习模型的预测准确率？
2. 在不同市场状态（趋势/震荡）下，市场结构信息的贡献度是否存在差异？
3. 与当前主流时间序列模型（LSTM、Transformer、PatchTST）相比，MST 的优势有多大？

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Market Structure Transformer                │
│                                                             │
│  ┌──────────────┐    ┌────────────────────┐                │
│  │ Price Series │    │ Market Structure   │                │
│  │  (OHLCV)     │    │  Features          │                │
│  └──────┬───────┘    │  - Trend State     │                │
│         │            │  - S/R Distance    │                │
│         │            │  - Volume Profile  │                │
│         │            │  - Volatility Reg. │                │
│         │            │  - Price Position  │                │
│         │            └────────┬───────────┘                │
│         ▼                     ▼                             │
│  ┌──────────────┐    ┌────────────────────┐                │
│  │ Price Token  │    │ Context Token      │                │
│  │ Embedding    │    │ Embedding          │                │
│  └──────┬───────┘    └────────┬───────────┘                │
│         │                     │                             │
│         ▼                     ▼                             │
│  ┌─────────────────────────────────────────┐               │
│  │      Cross-Attention Fusion Layer       │               │
│  │  (Price attends to Market Structure)    │               │
│  └─────────────────┬───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │     Transformer Encoder (N layers)      │               │
│  └─────────────────┬───────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│  ┌─────────────────────────────────────────┐               │
│  │         Prediction Head                 │               │
│  │   Classification: Up/Down               │               │
│  │   Regression: Next-day Return           │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

## Models Compared

| Model | Type | Description |
|-------|------|-------------|
| **LSTM** | Baseline | 2-layer LSTM with dropout |
| **GRU** | Baseline | 2-layer GRU with dropout |
| **TCN** | Baseline | Temporal Convolutional Network |
| **Transformer** | Baseline | Vanilla Transformer encoder |
| **PatchTST** | Strong baseline | Patch-based Time Series Transformer |
| **MST (Ours)** | Proposed | Market Structure Transformer with cross-attention context fusion |

## Datasets

- **CSI 300 成分股** (中国 A 股市场): 通过 AKShare 下载，日线数据
- **S&P 500 成分股** (美国市场): 通过 yfinance 下载，日线数据
- 时间跨度: 2015-01-01 ~ 2025-12-31
- 训练集/验证集/测试集: 按时间划分 (70%/15%/15%)

## Evaluation Metrics

### Classification (涨跌预测)
- Accuracy, Precision, Recall, F1-Score
- Matthews Correlation Coefficient (MCC)
- AUC-ROC

### Regression (收益率预测)
- MSE, MAE
- Information Coefficient (IC): Rank correlation between predicted and actual returns
- IC Information Ratio (ICIR): IC的均值/标准差

### Trading Performance (回测)
- Annualized Return, Annualized Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio

## Project Structure

```
project/
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── config.py                   # 配置与超参数
├── data/
│   ├── __init__.py
│   ├── download.py             # 数据下载 (yfinance / akshare)
│   ├── features.py             # 市场结构特征工程
│   └── dataset.py              # PyTorch Dataset & DataLoader
├── models/
│   ├── __init__.py
│   ├── lstm.py                 # LSTM / GRU 基线
│   ├── tcn.py                  # TCN 基线
│   ├── transformer.py          # Vanilla Transformer 基线
│   ├── patchtst.py             # PatchTST 基线
│   └── mst.py                  # Market Structure Transformer (核心创新)
├── utils/
│   ├── __init__.py
│   └── metrics.py              # 评估指标
├── train.py                    # 训练入口
└── evaluate.py                 # 评估与可视化
```

## Quick Start

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据
python -m data.download --market us --start 2015-01-01 --end 2025-12-31

# 3. 训练模型 (示例: 训练 MST)
python train.py --model mst --epochs 100 --batch_size 64

# 4. 训练所有基线模型
python train.py --model lstm --epochs 100
python train.py --model gru --epochs 100
python train.py --model tcn --epochs 100
python train.py --model transformer --epochs 100
python train.py --model patchtst --epochs 100

# 5. 评估与对比
python evaluate.py --models lstm gru tcn transformer patchtst mst
```

## Experimental Plan

### Experiment 1: Main Comparison
在 S&P 500 和 CSI 300 数据集上对比所有模型的分类和回归性能。

### Experiment 2: Ablation Study
- MST without cross-attention (直接拼接特征)
- MST without market structure features (仅价格)
- MST with different context feature subsets

### Experiment 3: Market Regime Analysis
将测试集按市场状态分层（牛市/熊市/震荡），分析各模型在不同状态下的表现。

### Experiment 4: Trading Simulation
基于模型预测信号构建简单多空策略，比较风险调整后收益。

## Timeline

| 阶段 | 时间 | 任务 |
|------|------|------|
| Week 1-2 | 3月下旬 | 数据收集、特征工程、基线模型实现 |
| Week 3-4 | 4月上旬 | MST 模型实现与调试 |
| Week 5-6 | 4月中下旬 | 实验运行、消融实验 |
| Week 7 | 5月初 | 论文撰写 (NeurIPS format) |
| Week 8 | 5月9日前 | 视频录制、最终提交 |

## References

1. Zhou, H., et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting." AAAI 2021.
2. Nie, Y., et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023. (PatchTST)
3. Xu, M., et al. "Stock Movement Prediction from Tweets and Historical Prices." ACL 2018.
4. Ding, X., et al. "Deep Learning for Event-Driven Stock Prediction." IJCAI 2015.
5. Bao, W., et al. "A deep learning framework for financial time series using stacked autoencoders and long-short term memory." PLOS ONE 2017.
