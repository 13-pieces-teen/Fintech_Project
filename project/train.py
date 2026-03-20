"""
Training script for stock movement prediction models.

Usage:
    python -m project.train --model mst --epochs 100 --batch_size 64
    python -m project.train --model lstm --epochs 100
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, DataConfig, ModelConfig, TrainConfig
from data.download import download_us_data
from data.dataset import create_dataloaders
from models import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for price, context, label in tqdm(loader, desc="  Train", leave=False):
        price = price.to(device)
        context = context.to(device)
        label = label.to(device)

        logits = model(price, context).squeeze(-1)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * price.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == label).sum().item()
        total_samples += price.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for price, context, label in tqdm(loader, desc="  Eval ", leave=False):
        price = price.to(device)
        context = context.to(device)
        label = label.to(device)

        logits = model(price, context).squeeze(-1)
        loss = criterion(logits, label)

        total_loss += loss.item() * price.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == label).sum().item()
        total_samples += price.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    parser = argparse.ArgumentParser(description="Train stock movement predictor")
    parser.add_argument("--model", type=str, default="mst",
                        choices=["lstm", "gru", "tcn", "transformer", "patchtst", "mst"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--market", type=str, default="us", choices=["us", "cn"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config()
    cfg.model.name = args.model
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.data.seq_len = args.seq_len
    cfg.data.market = args.market
    cfg.train.seed = args.seed

    set_seed(cfg.train.seed)

    device = torch.device(
        cfg.train.device if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Model:  {cfg.model.name}")

    print("\n--- Downloading / Loading data ---")
    if cfg.data.market == "us":
        all_data = download_us_data(
            cfg.data.tickers_us, cfg.data.start_date, cfg.data.end_date,
        )
    else:
        from data.download import download_cn_data
        all_data = download_cn_data(
            cfg.data.tickers_cn, cfg.data.start_date, cfg.data.end_date,
        )

    print(f"Loaded {len(all_data)} tickers")

    train_loader, val_loader, test_loader = create_dataloaders(
        all_data,
        seq_len=cfg.data.seq_len,
        pred_len=cfg.data.pred_len,
        label_type=cfg.data.label_type,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        batch_size=cfg.train.batch_size,
    )

    if train_loader is None:
        print("ERROR: No training data available. Check data download.")
        return

    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader) if val_loader else 0}, "
          f"Test batches: {len(test_loader) if test_loader else 0}")

    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)

    os.makedirs(cfg.train.save_dir, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>10} | {'Val Acc':>9} | {'LR':>10}")
    print(f"{'='*60}")

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )

        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>10.4f} | {val_acc:>8.2%} | {lr:>10.2e}  ({elapsed:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(cfg.train.save_dir, f"{cfg.model.name}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": cfg,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best val acc: {best_val_acc:.2%} at epoch {best_epoch}")
                break

    print(f"\n--- Training complete ---")
    print(f"Best val accuracy: {best_val_acc:.2%} at epoch {best_epoch}")

    if test_loader:
        ckpt_path = os.path.join(cfg.train.save_dir, f"{cfg.model.name}_best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test accuracy: {test_acc:.2%}, Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
