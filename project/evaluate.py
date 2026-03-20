"""
Evaluation and comparison script for all models.

Usage:
    python -m project.evaluate --models lstm gru tcn transformer patchtst mst
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data.download import download_us_data
from data.dataset import create_dataloaders
from models import build_model
from utils.metrics import compute_classification_metrics, compute_trading_metrics


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run model on a DataLoader and collect all predictions + labels."""
    model.eval()
    all_probs, all_labels = [], []

    for price, context, label in tqdm(loader, desc="  Predict", leave=False):
        price = price.to(device)
        context = context.to(device)

        logits = model(price, context).squeeze(-1)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(label.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def evaluate_all_models(model_names, test_loader, device, save_dir):
    """Evaluate all trained models and print comparison table."""
    results = {}

    for name in model_names:
        ckpt_path = os.path.join(save_dir, f"{name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: checkpoint not found for {name}, skipping.")
            continue

        print(f"\nEvaluating: {name}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        cfg.model.name = name
        model = build_model(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        probs, labels = collect_predictions(model, test_loader, device)
        cls_metrics = compute_classification_metrics(labels, probs)

        results[name] = {
            "probs": probs,
            "labels": labels,
            **cls_metrics,
        }

    return results


def print_comparison_table(results):
    """Print formatted comparison table."""
    if not results:
        print("No results to display.")
        return

    header = f"{'Model':<15} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} " \
             f"{'F1':>6} {'MCC':>6} {'AUC':>6}"
    print(f"\n{'='*65}")
    print(header)
    print(f"{'='*65}")

    for name, m in results.items():
        print(f"{name:<15} {m['accuracy']:>8.2%} {m['precision']:>9.2%} "
              f"{m['recall']:>8.2%} {m['f1']:>6.3f} {m['mcc']:>6.3f} "
              f"{m['auc_roc']:>6.3f}")
    print(f"{'='*65}")


def plot_comparison(results, save_path="project/results"):
    """Generate comparison plots."""
    os.makedirs(save_path, exist_ok=True)

    if not results:
        return

    models = list(results.keys())
    metrics_to_plot = ["accuracy", "f1", "mcc", "auc_roc"]
    metric_labels = ["Accuracy", "F1-Score", "MCC", "AUC-ROC"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 4))

    for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
        values = [results[m][metric] for m in models]
        colors = ["#e74c3c" if m == "mst" else "#3498db" for m in models]
        bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Comparison on Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved comparison plot to {save_path}/model_comparison.png")


def plot_ablation(results, save_path="project/results"):
    """
    If MST variants exist, plot ablation study:
    - mst (full model)
    - mst_no_cross_attn (without cross-attention)
    - mst_no_context (without market structure features)
    """
    ablation_models = [m for m in results if m.startswith("mst")]
    if len(ablation_models) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["accuracy", "f1", "mcc"]
    x = np.arange(len(metrics))
    width = 0.8 / len(ablation_models)

    for i, name in enumerate(ablation_models):
        values = [results[name][m] for m in metrics]
        ax.bar(x + i * width, values, width, label=name, alpha=0.8)

    ax.set_xticks(x + width * (len(ablation_models) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "F1-Score", "MCC"])
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: MST Components")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "ablation_study.png"), dpi=150, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--models", nargs="+",
                        default=["lstm", "gru", "tcn", "transformer", "patchtst", "mst"])
    parser.add_argument("--market", type=str, default="us")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.data.market = args.market

    print("--- Loading data ---")
    if cfg.data.market == "us":
        all_data = download_us_data(
            cfg.data.tickers_us, cfg.data.start_date, cfg.data.end_date,
        )
    else:
        from data.download import download_cn_data
        all_data = download_cn_data(
            cfg.data.tickers_cn, cfg.data.start_date, cfg.data.end_date,
        )

    _, _, test_loader = create_dataloaders(
        all_data,
        seq_len=cfg.data.seq_len,
        pred_len=cfg.data.pred_len,
        label_type=cfg.data.label_type,
        batch_size=args.batch_size,
    )

    if test_loader is None:
        print("ERROR: No test data available.")
        return

    print(f"\n--- Evaluating {len(args.models)} models ---")
    results = evaluate_all_models(args.models, test_loader, device, cfg.train.save_dir)

    print_comparison_table(results)
    plot_comparison(results)
    plot_ablation(results)


if __name__ == "__main__":
    main()
