"""Data download utilities for US (yfinance) and CN (akshare) markets."""

import os
import argparse
from pathlib import Path

import pandas as pd


def download_us_data(
    tickers: list[str],
    start: str,
    end: str,
    save_dir: str = "project/data/raw/us",
) -> dict[str, pd.DataFrame]:
    """Download US stock daily OHLCV data via yfinance."""
    import yfinance as yf

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    data = {}

    for ticker in tickers:
        path = os.path.join(save_dir, f"{ticker}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            print(f"Downloading {ticker} ...")
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                print(f"  WARNING: no data for {ticker}, skipping.")
                continue
            df.columns = [c.lower() for c in df.columns]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["open", "high", "low", "close", "volume"]]
            df.to_csv(path)
            print(f"  Saved {len(df)} rows → {path}")
        data[ticker] = df

    return data


def download_cn_data(
    tickers: list[str],
    start: str,
    end: str,
    save_dir: str = "project/data/raw/cn",
) -> dict[str, pd.DataFrame]:
    """Download Chinese A-share daily OHLCV data via akshare."""
    import akshare as ak

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    data = {}

    for ticker in tickers:
        path = os.path.join(save_dir, f"{ticker}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            print(f"Downloading {ticker} ...")
            try:
                df = ak.stock_zh_a_hist(
                    symbol=ticker,
                    period="daily",
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                    adjust="qfq",
                )
            except Exception as e:
                print(f"  WARNING: failed for {ticker}: {e}")
                continue
            df = df.rename(columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
            df.to_csv(path)
            print(f"  Saved {len(df)} rows → {path}")
        data[ticker] = df

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data")
    parser.add_argument("--market", type=str, default="us", choices=["us", "cn"])
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    args = parser.parse_args()

    from config import DataConfig
    cfg = DataConfig(market=args.market, start_date=args.start, end_date=args.end)

    if args.market == "us":
        download_us_data(cfg.tickers_us, cfg.start_date, cfg.end_date)
    else:
        download_cn_data(cfg.tickers_cn, cfg.start_date, cfg.end_date)
