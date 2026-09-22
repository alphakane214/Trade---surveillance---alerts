"""
Market manipulation detection algorithms.

Implements two classic surveillance patterns:
    1. Spoofing  - large orders placed then cancelled quickly without
                   execution, used to create a false impression of supply/demand.
    2. Wash trading - an account (or linked accounts) trades with itself,
                   creating artificial volume without real economic risk transfer.
"""

import pandas as pd
import numpy as np


def detect_spoofing(
    df: pd.DataFrame,
    quantity_zscore_threshold: float = 2.5,
    cancel_only: bool = True,
) -> pd.DataFrame:
    """
    Flags orders that are:
        - Cancelled (not filled), AND
        - Abnormally large relative to that symbol's typical order size
          (z-score based outlier detection)

    Returns the input dataframe with an added `flagged_spoofing` boolean column.
    """
    df = df.copy()
    df["flagged_spoofing"] = False

    for symbol, group in df.groupby("symbol"):
        mean_qty = group["quantity"].mean()
        std_qty = group["quantity"].std()
        if std_qty == 0 or np.isnan(std_qty):
            continue

        zscores = (group["quantity"] - mean_qty) / std_qty
        is_large = zscores > quantity_zscore_threshold
        is_cancelled = group["status"] == "CANCELLED" if cancel_only else True

        flagged_idx = group[is_large & is_cancelled].index
        df.loc[flagged_idx, "flagged_spoofing"] = True

    return df


def detect_wash_trading(
    df: pd.DataFrame,
    time_window_seconds: int = 60,
    price_tolerance_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Flags pairs of trades that look like wash trades:
        - Opposite sides (one BUY, one SELL) on the same symbol
        - Near-identical price (within price_tolerance_pct)
        - Near-identical quantity
        - Executed within a short time window of each other
        - Different accounts (a same-account match would be an even more
          obvious case, also caught here)

    This is a simplified pairwise matcher intended for demonstration; a
    production system would use a more scalable matching engine.

    Returns the input dataframe with an added `flagged_wash_trade` column.
    """
    df = df.copy()
    df["flagged_wash_trade"] = False
    df = df.sort_values("timestamp").reset_index(drop=True)

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("timestamp")
        buys = group[group["side"] == "BUY"]
        sells = group[group["side"] == "SELL"]

        for sell_idx, sell in sells.iterrows():
            window_start = sell["timestamp"] - pd.Timedelta(seconds=time_window_seconds)
            window_end = sell["timestamp"] + pd.Timedelta(seconds=time_window_seconds)

            candidates = buys[
                (buys["timestamp"] >= window_start)
                & (buys["timestamp"] <= window_end)
                & (buys["account_id"] != sell["account_id"])
            ]

            if candidates.empty:
                continue

            price_diff_pct = (candidates["price"] - sell["price"]).abs() / sell["price"] * 100
            qty_diff_pct = (candidates["quantity"] - sell["quantity"]).abs() / sell["quantity"] * 100

            matches = candidates[
                (price_diff_pct <= price_tolerance_pct) & (qty_diff_pct <= price_tolerance_pct * 10)
            ]

            if not matches.empty:
                df.loc[sell_idx, "flagged_wash_trade"] = True
                df.loc[matches.index, "flagged_wash_trade"] = True

    return df


def run_manipulation_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper running both detectors and returning combined results."""
    df = detect_spoofing(df)
    df = detect_wash_trading(df)
    return df


def summarize_detection(df: pd.DataFrame) -> dict:
    """Compare flagged results against ground-truth labels (for synthetic data validation)."""
    spoof_tp = ((df["flagged_spoofing"]) & (df["is_spoofing"])).sum()
    spoof_fp = ((df["flagged_spoofing"]) & (~df["is_spoofing"])).sum()
    spoof_fn = ((~df["flagged_spoofing"]) & (df["is_spoofing"])).sum()

    wash_tp = ((df["flagged_wash_trade"]) & (df["is_wash_trade"])).sum()
    wash_fp = ((df["flagged_wash_trade"]) & (~df["is_wash_trade"])).sum()
    wash_fn = ((~df["flagged_wash_trade"]) & (df["is_wash_trade"])).sum()

    return {
        "spoofing": {"true_positives": int(spoof_tp), "false_positives": int(spoof_fp), "false_negatives": int(spoof_fn)},
        "wash_trading": {"true_positives": int(wash_tp), "false_positives": int(wash_fp), "false_negatives": int(wash_fn)},
    }
