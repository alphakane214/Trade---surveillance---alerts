import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import pytest

from data_generation.generate_trades import (
    generate_baseline_orders,
    inject_spoofing,
    inject_wash_trading,
    inject_insider_trading,
)
from detection.market_manipulation import detect_spoofing, detect_wash_trading
from detection.insider_trading import detect_pre_event_accumulation


@pytest.fixture(scope="module")
def dataset():
    df = generate_baseline_orders(n=1000)
    df = inject_spoofing(df, n_events=10)
    df = inject_wash_trading(df, n_pairs=3, trades_per_pair=5)
    df, event_date = inject_insider_trading(df, n_accounts=3)
    return df, event_date


def test_spoofing_detection_catches_injected_events(dataset):
    df, _ = dataset
    result = detect_spoofing(df)
    caught = result[result["is_spoofing"] & result["flagged_spoofing"]]
    assert len(caught) == df["is_spoofing"].sum(), "Should catch all injected spoofing events"


def test_spoofing_detection_low_false_positive_rate(dataset):
    df, _ = dataset
    result = detect_spoofing(df)
    false_positives = result[result["flagged_spoofing"] & ~result["is_spoofing"]]
    total_normal = (~df["is_spoofing"]).sum()
    fp_rate = len(false_positives) / total_normal
    assert fp_rate < 0.05, "False positive rate should be low on normal baseline activity"


def test_wash_trading_detection_catches_injected_pairs(dataset):
    df, _ = dataset
    result = detect_wash_trading(df)
    caught = result[result["is_wash_trade"] & result["flagged_wash_trade"]]
    assert len(caught) > 0, "Should catch at least some injected wash trade legs"


def test_insider_detection_flags_true_insider_accounts(dataset):
    df, event_date = dataset
    symbol = df[df["is_insider"]]["symbol"].iloc[0]
    flagged = detect_pre_event_accumulation(df, event_date, symbol=symbol)
    flagged_accounts = set(flagged[flagged["flagged_insider"]]["account_id"])
    true_accounts = set(df[df["is_insider"]]["account_id"])
    assert len(flagged_accounts & true_accounts) > 0, "Should flag at least one true insider account"


def test_no_detection_on_clean_data():
    """Sanity check: baseline-only data (no injected anomalies) should
    produce very few or no flags."""
    df = generate_baseline_orders(n=500)
    result = detect_spoofing(df)
    flagged_rate = result["flagged_spoofing"].mean()
    assert flagged_rate < 0.05, "Clean baseline data should rarely trigger spoofing flags"
