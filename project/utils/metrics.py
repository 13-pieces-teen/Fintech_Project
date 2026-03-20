"""Evaluation metrics for stock movement prediction."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth binary labels (0/1)
        y_prob: Predicted probabilities
        threshold: Classification threshold
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = 0.0

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def compute_trading_metrics(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> dict:
    """
    Simulate a long/short strategy and compute trading performance metrics.

    Strategy: go long when model predicts up (>0.5), short when down (<0.5).

    Args:
        predictions: Model probability outputs
        actual_returns: Actual next-day returns
        risk_free_rate: Annual risk-free rate
        trading_days: Trading days per year
    """
    positions = np.where(predictions > 0.5, 1.0, -1.0)
    strategy_returns = positions * actual_returns

    cumulative = np.cumprod(1 + strategy_returns)
    total_return = cumulative[-1] - 1.0

    ann_return = (1 + total_return) ** (trading_days / len(strategy_returns)) - 1
    ann_vol = np.std(strategy_returns) * np.sqrt(trading_days)
    sharpe = (ann_return - risk_free_rate) / (ann_vol + 1e-8)

    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns)

    calmar = ann_return / (abs(max_drawdown) + 1e-8)

    win_trades = strategy_returns > 0
    win_rate = np.mean(win_trades)

    ic = np.corrcoef(predictions, actual_returns)[0, 1] if len(predictions) > 1 else 0.0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "information_coefficient": ic,
        "cumulative_returns": cumulative.tolist(),
    }
