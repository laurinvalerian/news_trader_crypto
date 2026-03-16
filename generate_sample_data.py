"""
generate_sample_data.py
========================
Generate synthetic sample data for testing the news_strangle_backtester.

This script creates:
  1. data/economic_calendar.csv  – synthetic USD High/Medium impact events
  2. data/btcusdt_1m.csv         – synthetic 1-minute BTC OHLCV data

The generated data is purely synthetic and intended ONLY for testing the
backtester logic. For real backtesting results, download actual datasets
from Kaggle or Binance (see README).
"""

import os
import random
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED   = 42
DATA_DIR      = Path("data")
START_DATE    = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
END_DATE      = dt.datetime(2026, 3, 15, tzinfo=dt.timezone.utc)
INITIAL_PRICE = 16_500.0   # BTC price at 2023-01-01 approx.


def generate_ohlcv(
    start: dt.datetime = START_DATE,
    end: dt.datetime = END_DATE,
    initial_price: float = INITIAL_PRICE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate synthetic 1-minute BTC OHLCV data using a random walk.

    The price drifts upwards slightly and has realistic-ish volatility
    with occasional news-spike injections.
    """
    rng = np.random.default_rng(seed)

    # Generate minute-by-minute timestamps
    timestamps = pd.date_range(start=start, end=end, freq="min", tz="UTC")
    n = len(timestamps)

    # Random walk with slight upward drift
    drift     = 0.000001    # tiny upward bias per minute
    vol       = 0.0003      # per-minute volatility
    returns   = rng.normal(drift, vol, n)
    log_prices = np.log(initial_price) + np.cumsum(returns)
    closes     = np.exp(log_prices)

    # Generate OHLC from close prices
    noise = rng.uniform(0.9995, 1.0005, (n, 3))   # small noise for O/H/L
    opens  = closes * noise[:, 0]
    highs  = np.maximum(closes, opens) * rng.uniform(1.0000, 1.0015, n)
    lows   = np.minimum(closes, opens) * rng.uniform(0.9985, 1.0000, n)

    # Inject larger moves around certain "event" times to make
    # the backtest more interesting
    event_indices = rng.choice(n, size=min(200, n // 1000), replace=False)
    for idx in event_indices:
        spike = rng.choice([-1, 1]) * rng.uniform(200, 800)
        end_idx = min(idx + 60, n)
        for i in range(idx, end_idx):
            factor = spike * (1 - (i - idx) / (end_idx - idx)) / closes[i]
            closes[i]  *= (1 + factor * 0.05)
            highs[i]    = max(highs[i], closes[i] * 1.001)
            lows[i]     = min(lows[i], closes[i] * 0.999)

    volumes = rng.uniform(0.5, 50.0, n)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open":      np.round(opens, 2),
        "high":      np.round(highs, 2),
        "low":       np.round(lows, 2),
        "close":     np.round(closes, 2),
        "volume":    np.round(volumes, 4),
    })

    return df


def generate_calendar(
    start: dt.datetime = START_DATE,
    end: dt.datetime = END_DATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate a synthetic economic calendar with USD events.

    Produces ~3 events per week (roughly matching real-world frequency
    of major USD economic releases like NFP, CPI, FOMC, etc.).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    events = []
    current = start
    event_names = [
        "Non-Farm Payrolls", "CPI m/m", "CPI y/y", "Core CPI",
        "FOMC Statement", "Fed Interest Rate Decision", "GDP q/q",
        "Retail Sales m/m", "Unemployment Claims", "PPI m/m",
        "ISM Manufacturing PMI", "ISM Services PMI", "Consumer Confidence",
        "Durable Goods Orders", "ADP Employment", "PCE Price Index",
        "Building Permits", "New Home Sales", "Trade Balance",
    ]

    while current < end:
        # ~3 events per week → on average every 2-3 days
        gap_hours = rng.integers(36, 84)
        current += dt.timedelta(hours=int(gap_hours))
        if current >= end:
            break

        # Set event time to typical US release times (13:30 or 15:00 UTC)
        hour = rng.choice([13, 14, 15, 18, 19])
        minute = rng.choice([0, 30])
        evt_time = current.replace(hour=hour, minute=minute, second=0, microsecond=0)

        impact = rng.choice(["High", "Medium"], p=[0.35, 0.65])
        name   = rng.choice(event_names)

        events.append({
            "time":     evt_time.strftime("%Y-%m-%d %H:%M:%S"),
            "currency": "USD",
            "impact":   impact,
            "event":    name,
        })

    # Add a few non-USD events (should be filtered out)
    for _ in range(50):
        gap_hours = rng.integers(36, 168)
        t = start + dt.timedelta(hours=int(gap_hours * rng.random() * 20))
        if t < end:
            events.append({
                "time":     t.strftime("%Y-%m-%d %H:%M:%S"),
                "currency": rng.choice(["EUR", "GBP", "JPY", "CAD"]),
                "impact":   rng.choice(["High", "Medium", "Low"]),
                "event":    "Foreign Event",
            })

    # Add some Low impact USD events (should also be filtered out)
    for _ in range(30):
        t = start + dt.timedelta(hours=int(rng.integers(24, 20000)))
        if t < end:
            events.append({
                "time":     t.strftime("%Y-%m-%d %H:%M:%S"),
                "currency": "USD",
                "impact":   "Low",
                "event":    "Low Impact Event",
            })

    df = pd.DataFrame(events)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def main():
    """Generate and save sample data to the data/ directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic OHLCV data (this may take a moment)...")
    ohlcv = generate_ohlcv()
    ohlcv_path = DATA_DIR / "btcusdt_1m.csv"
    ohlcv.to_csv(ohlcv_path, index=False)
    print(f"  → Saved {len(ohlcv)} bars to {ohlcv_path}")

    print("Generating synthetic economic calendar...")
    cal = generate_calendar()
    cal_path = DATA_DIR / "economic_calendar.csv"
    cal.to_csv(cal_path, index=False)
    print(f"  → Saved {len(cal)} events to {cal_path}")

    print("\nDone! You can now run: python news_strangle_backtester.py")


if __name__ == "__main__":
    main()
