"""
End-to-end demo: generates synthetic data, runs all detectors, and produces
summary charts saved to notebooks/output/.

Run from the project root:
    python notebooks/run_full_report.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_generation.generate_trades import generate_full_dataset
from detection.market_manipulation import run_manipulation_detection, summarize_detection
from detection.insider_trading import detect_pre_event_accumulation, summarize_insider_detection

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("Generating synthetic dataset...")
    df, event_date = generate_full_dataset(
        output_path=os.path.join(os.path.dirname(__file__), "..", "data", "raw", "trades.csv")
    )

    print("\nRunning market manipulation detection...")
    result = run_manipulation_detection(df)
    manip_summary = summarize_detection(result)
    print(manip_summary)

    print("\nRunning insider trading detection...")
    insider_symbol = df[df["is_insider"]]["symbol"].iloc[0]
    flagged_accounts = detect_pre_event_accumulation(df, event_date, symbol=insider_symbol)
    insider_summary = summarize_insider_detection(df, flagged_accounts)
    print(insider_summary)

    # --- Chart 1: Spoofing - order quantity distribution, flagged vs not ---
    fig, ax = plt.subplots(figsize=(8, 5))
    normal = result[~result["flagged_spoofing"]]["quantity"]
    flagged = result[result["flagged_spoofing"]]["quantity"]
    ax.hist(normal, bins=40, alpha=0.6, label="Normal orders")
    ax.hist(flagged, bins=40, alpha=0.8, label="Flagged as spoofing", color="crimson")
    ax.set_xlabel("Order quantity")
    ax.set_ylabel("Count")
    ax.set_title("Spoofing Detection: Order Size Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "spoofing_distribution.png"), dpi=150)
    plt.close(fig)

    # --- Chart 2: Wash trading - flagged trades over time for affected symbol ---
    wash_symbol = df[df["is_wash_trade"]]["symbol"].iloc[0]
    subset = result[result["symbol"] == wash_symbol].sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(subset["timestamp"], subset["price"],
               c=subset["flagged_wash_trade"].map({True: "crimson", False: "steelblue"}),
               s=subset["flagged_wash_trade"].map({True: 40, False: 10}))
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(f"Wash Trading Detection: {wash_symbol} (red = flagged)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "wash_trading_timeline.png"), dpi=150)
    plt.close(fig)

    # --- Chart 3: Insider trading - accumulation before event + price jump ---
    insider_df = df[df["symbol"] == insider_symbol].sort_values("timestamp").copy()
    insider_df["date"] = insider_df["timestamp"].dt.floor("h")
    hourly = insider_df.groupby("date").agg(avg_price=("price", "mean"), volume=("quantity", "sum")).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(hourly["date"], hourly["avg_price"], color="steelblue")
    ax1.axvline(event_date, color="crimson", linestyle="--", label="News event")
    ax1.set_ylabel("Avg price")
    ax1.set_title(f"Insider Trading Pattern: {insider_symbol}")
    ax1.legend()

    ax2.bar(hourly["date"], hourly["volume"], width=0.03, color="darkorange")
    ax2.axvline(event_date, color="crimson", linestyle="--")
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "insider_trading_pattern.png"), dpi=150)
    plt.close(fig)

    # --- Summary text file ---
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write("TRADE SURVEILLANCE DETECTION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total orders analyzed: {len(df)}\n\n")
        f.write("Market Manipulation Detection:\n")
        f.write(f"  Spoofing     - TP: {manip_summary['spoofing']['true_positives']}, "
                f"FP: {manip_summary['spoofing']['false_positives']}, "
                f"FN: {manip_summary['spoofing']['false_negatives']}\n")
        f.write(f"  Wash trading - TP: {manip_summary['wash_trading']['true_positives']}, "
                f"FP: {manip_summary['wash_trading']['false_positives']}, "
                f"FN: {manip_summary['wash_trading']['false_negatives']}\n\n")
        f.write("Insider Trading Detection:\n")
        f.write(f"  TP: {insider_summary['true_positives']}, "
                f"FP: {insider_summary['false_positives']}, "
                f"FN: {insider_summary['false_negatives']}\n")
        f.write(f"  True insider accounts: {insider_summary['true_insider_accounts']}\n")
        f.write(f"  Flagged accounts:      {insider_summary['flagged_accounts']}\n")

    print(f"\nCharts and summary saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
