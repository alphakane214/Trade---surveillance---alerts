"""
Insider trading detection.

Classic pattern: a small cluster of accounts accumulates an abnormally
large position in a symbol shortly before a material news event (earnings,
M&A announcement, etc.), then the price jumps once the news becomes public.

This module implements a simplified version of that pattern:
    1. Identify a price/volume "jump" (the event date) via rolling statistics
    2. Look backward from that date for accounts with abnormal accumulation
       relative to their own historical baseline and relative to the market
"""

import pandas as pd
import numpy as np


def detect_price_volume_events(
    df: pd.DataFrame,
    price_jump_zscore: float = 2.0,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Aggregates trades by symbol/day and flags days with abnormal price
    or volume jumps relative to that symbol's trailing history.

    Returns a per-symbol/day dataframe with z-scores and a `flagged_event` column.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.floor(freq)

    daily = (
        df.groupby(["symbol", "date"])
        .agg(avg_price=("price", "mean"), volume=("quantity", "sum"), n_orders=("order_id", "count"))
        .reset_index()
    )

    daily["price_pct_change"] = daily.groupby("symbol")["avg_price"].pct_change()
    daily["volume_zscore"] = daily.groupby("symbol")["volume"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    price_change_std = daily.groupby("symbol")["price_pct_change"].transform("std")
    daily["price_zscore"] = daily["price_pct_change"] / price_change_std

    daily["flagged_event"] = (daily["price_zscore"].abs() > price_jump_zscore) | (
        daily["volume_zscore"] > price_jump_zscore
    )

    return daily


def detect_pre_event_accumulation(
    df: pd.DataFrame,
    event_date: pd.Timestamp,
    symbol: str,
    lookback_hours: int = 48,
    account_qty_zscore_threshold: float = 1.5,
    min_orders: int = 2,
) -> pd.DataFrame:
    """
    Given a known (or detected) event date, looks backward `lookback_hours`
    and flags accounts whose order quantity for that symbol is abnormally
    large relative to all other accounts' activity in the same pre-event window.

    Returns a per-account summary dataframe sorted by suspicion score, with
    a `flagged_insider` boolean column.
    """
    df = df.copy()
    window_start = event_date - pd.Timedelta(hours=lookback_hours)

    pre_event = df[
        (df["symbol"] == symbol)
        & (df["timestamp"] >= window_start)
        & (df["timestamp"] < event_date)
        & (df["side"] == "BUY")
    ]

    if pre_event.empty:
        return pd.DataFrame()

    by_account = (
        pre_event.groupby("account_id")
        .agg(total_qty=("quantity", "sum"), n_orders=("order_id", "count"), avg_price=("price", "mean"))
        .reset_index()
    )

    mean_qty = by_account["total_qty"].mean()
    std_qty = by_account["total_qty"].std()
    by_account["qty_zscore"] = (by_account["total_qty"] - mean_qty) / std_qty if std_qty > 0 else 0

    by_account["flagged_insider"] = (
        (by_account["qty_zscore"] > account_qty_zscore_threshold)
        & (by_account["n_orders"] >= min_orders)
    )

    return by_account.sort_values("qty_zscore", ascending=False).reset_index(drop=True)


def summarize_insider_detection(df: pd.DataFrame, flagged_accounts: pd.DataFrame) -> dict:
    """Compare flagged accounts against ground-truth is_insider label (for synthetic validation)."""
    true_insider_accounts = set(df[df["is_insider"]]["account_id"].unique())
    flagged = set(flagged_accounts[flagged_accounts["flagged_insider"]]["account_id"])

    tp = len(flagged & true_insider_accounts)
    fp = len(flagged - true_insider_accounts)
    fn = len(true_insider_accounts - flagged)

    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "true_insider_accounts": sorted(true_insider_accounts), "flagged_accounts": sorted(flagged)}
