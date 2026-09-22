"""
Synthetic trade & order data generator for trade surveillance demos.

Generates:
    - Baseline "normal" order/trade activity for a set of accounts and symbols
    - Injected spoofing patterns (large orders placed then cancelled without execution)
    - Injected wash trading patterns (linked accounts trading with each other)
    - Injected insider trading patterns (abnormal volume/price before a "news event")

Ground truth labels are preserved in a separate column so detection algorithms
can be validated against known answers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

SYMBOLS = ["ACME", "GLOBEX", "INITECH", "UMBRELLA", "WAYNE"]
ACCOUNTS = [f"ACC{i:04d}" for i in range(1, 51)]


def _random_timestamps(n, start, end):
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    ts = RNG.uniform(start_ts, end_ts, n)
    return sorted(pd.to_datetime(ts, unit="s"))


def generate_baseline_orders(n=5000, start=None, end=None):
    """Generate baseline (non-manipulative) order activity."""
    start = start or datetime(2025, 1, 1)
    end = end or datetime(2025, 1, 31)

    timestamps = _random_timestamps(n, start, end)
    symbols = RNG.choice(SYMBOLS, n)
    accounts = RNG.choice(ACCOUNTS, n)
    sides = RNG.choice(["BUY", "SELL"], n)
    prices = np.round(RNG.normal(100, 5, n).clip(1, None), 2)
    quantities = RNG.integers(10, 500, n)

    # Most orders execute; a small baseline cancel rate is normal market behavior
    statuses = RNG.choice(
        ["FILLED", "CANCELLED"], n, p=[0.9, 0.1]
    )

    df = pd.DataFrame({
        "order_id": [f"O{i:07d}" for i in range(n)],
        "timestamp": timestamps,
        "account_id": accounts,
        "symbol": symbols,
        "side": sides,
        "price": prices,
        "quantity": quantities,
        "status": statuses,
        "is_spoofing": False,
        "is_wash_trade": False,
        "is_insider": False,
    })
    return df


def inject_spoofing(df, n_events=15, symbol="ACME"):
    """
    Inject spoofing pattern: an account places a large order far from the
    prevailing price, then cancels it within seconds, without execution.
    This is designed to move the visible order book without intent to trade.
    """
    rows = []
    base_time = df["timestamp"].min() + timedelta(days=2)

    for i in range(n_events):
        account = RNG.choice(ACCOUNTS)
        ts = base_time + timedelta(hours=int(RNG.integers(0, 600)))
        large_qty = int(RNG.integers(2000, 5000))  # much larger than baseline
        price = round(RNG.normal(100, 5), 2)

        rows.append({
            "order_id": f"SP{i:05d}",
            "timestamp": ts,
            "account_id": account,
            "symbol": symbol,
            "side": RNG.choice(["BUY", "SELL"]),
            "price": price,
            "quantity": large_qty,
            "status": "CANCELLED",
            "is_spoofing": True,
            "is_wash_trade": False,
            "is_insider": False,
        })

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def inject_wash_trading(df, n_pairs=5, trades_per_pair=8, symbol="GLOBEX"):
    """
    Inject wash trading pattern: two linked accounts repeatedly trade with
    each other at similar prices/quantities, creating artificial volume
    without real economic risk transfer.
    """
    rows = []
    base_time = df["timestamp"].min() + timedelta(days=5)

    for p in range(n_pairs):
        acc_a, acc_b = f"WASH_A{p}", f"WASH_B{p}"
        for t in range(trades_per_pair):
            ts = base_time + timedelta(hours=int(p * 20 + t * 2))
            price = round(100 + RNG.normal(0, 0.5), 2)
            qty = int(RNG.integers(100, 300))

            # A sells to B
            rows.append({
                "order_id": f"WA{p}{t:03d}",
                "timestamp": ts,
                "account_id": acc_a,
                "symbol": symbol,
                "side": "SELL",
                "price": price,
                "quantity": qty,
                "status": "FILLED",
                "is_spoofing": False,
                "is_wash_trade": True,
                "is_insider": False,
            })
            # B buys from A, near-identical terms, moments later
            rows.append({
                "order_id": f"WB{p}{t:03d}",
                "timestamp": ts + timedelta(seconds=int(RNG.integers(1, 30))),
                "account_id": acc_b,
                "symbol": symbol,
                "side": "BUY",
                "price": price,
                "quantity": qty,
                "status": "FILLED",
                "is_spoofing": False,
                "is_wash_trade": True,
                "is_insider": False,
            })

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def inject_insider_trading(df, symbol="INITECH", event_date=None, n_accounts=4):
    """
    Inject insider trading pattern: a small cluster of accounts trades an
    abnormally large volume of a specific symbol in the 48 hours before a
    simulated "news event" (e.g., earnings surprise), then the price jumps
    sharply on the event date itself.
    """
    event_date = event_date or (df["timestamp"].min() + timedelta(days=15))
    rows = []

    suspicious_accounts = [f"INSIDE{i}" for i in range(n_accounts)]

    # Pre-event: abnormal accumulation, well above baseline order sizes
    for i in range(30):
        acc = RNG.choice(suspicious_accounts)
        ts = event_date - timedelta(hours=int(RNG.integers(1, 48)))
        rows.append({
            "order_id": f"IN{i:05d}",
            "timestamp": ts,
            "account_id": acc,
            "symbol": symbol,
            "side": "BUY",
            "price": round(RNG.normal(100, 1), 2),  # price hasn't moved yet
            "quantity": int(RNG.integers(800, 1500)),  # abnormally large
            "status": "FILLED",
            "is_spoofing": False,
            "is_wash_trade": False,
            "is_insider": True,
        })

    # Event date: price jumps, everyone else starts trading normally at the new price
    for i in range(40):
        ts = event_date + timedelta(hours=int(RNG.integers(0, 24)))
        rows.append({
            "order_id": f"POST{i:05d}",
            "timestamp": ts,
            "account_id": RNG.choice(ACCOUNTS),
            "symbol": symbol,
            "side": RNG.choice(["BUY", "SELL"]),
            "price": round(RNG.normal(130, 3), 2),  # price jumped ~30%
            "quantity": int(RNG.integers(50, 400)),
            "status": "FILLED",
            "is_spoofing": False,
            "is_wash_trade": False,
            "is_insider": False,
        })

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True), event_date


def generate_full_dataset(output_path="data/raw/trades.csv"):
    df = generate_baseline_orders()
    df = inject_spoofing(df)
    df = inject_wash_trading(df)
    df, event_date = inject_insider_trading(df)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} orders -> {output_path}")
    print(f"  Spoofing events:      {df['is_spoofing'].sum()}")
    print(f"  Wash trade legs:      {df['is_wash_trade'].sum()}")
    print(f"  Insider trade orders: {df['is_insider'].sum()}")
    print(f"  Simulated news event date (INITECH): {event_date}")

    return df, event_date


if __name__ == "__main__":
    generate_full_dataset()
