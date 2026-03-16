"""
news_strangle_backtester.py
============================
Phase 1 – Standalone Backtester for the "Free Crypto News Strangle Strategy"
on BTCUSDT Perpetual Futures (Binance).

Period : 2023-01-01  →  2026-03-15  (or max available data)
Author : AutoNews Quant Team
Python : 3.10+

Strategy Rules (immutable):
  • Only High + Medium impact USD economic events.
  • 5 min before event → place strangle:
        Buy-Stop  = current_price + STRANGLE_OFFSET
        Sell-Stop = current_price − STRANGLE_OFFSET
  • Position size = exactly 1 % account risk (based on SL distance).
  • SL = 800 USDT (fixed).   TP = 6 000 USDT (fixed) → 1 : 7.5 RR.
  • Once one pending order triggers the other is cancelled immediately.
  • 60-second confirmation filter after trigger:
        Price must move ≥ CONFIRMATION_MIN in trigger direction.
        Pass → trade continues with SL / TP.
        Fail → close immediately (fakeout).
  • Slippage: +0.15 % … +0.50 % random on news-release candles.
  • No trailing, no partial close, no directional bias – purely neutral.
  • One trade per event only (no re-entry).
  • Max hold time = 2 h, then auto-close.

Data sources (CSV, user-provided):
  1. Economic calendar  → data/economic_calendar.csv
  2. BTC 1-min OHLCV    → data/btcusdt_1m.csv
"""

import os
import sys
import datetime as dt
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless backend for server envs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ─────────────────────────── STRATEGY CONSTANTS ─────────────────────────── #

STRANGLE_OFFSET      = 400.0    # USDT above / below current price
SL_DISTANCE           = 800.0    # fixed stop-loss in USDT
TP_DISTANCE           = 6000.0   # fixed take-profit in USDT  (RR 1:7.5)
ACCOUNT_RISK_PCT      = 0.01     # 1 % of equity risked per trade
INITIAL_EQUITY        = 10_000.0 # starting account balance in USDT
CONFIRMATION_WAIT_SEC = 60       # seconds to wait after trigger
CONFIRMATION_MIN      = 300.0    # minimum price move (USDT) to confirm
MAX_HOLD_SEC          = 7200     # 2 hours max hold time
SLIPPAGE_MIN_PCT      = 0.0015   # 0.15 % minimum slippage
SLIPPAGE_MAX_PCT      = 0.0050   # 0.50 % maximum slippage
MINUTES_BEFORE_EVENT  = 5        # place strangle N minutes before event
RANDOM_SEED           = 42       # reproducibility

# Backtest PASS criteria
PASS_EXPECTANCY_MIN   = 1.5      # minimum expectancy in R
PASS_MAX_DD_PCT       = 15.0     # maximum drawdown %
PASS_PF_MIN           = 1.8      # minimum profit factor

# File paths  (relative to project root)
DATA_DIR              = Path("data")
CALENDAR_CSV          = DATA_DIR / "economic_calendar.csv"
OHLCV_CSV             = DATA_DIR / "btcusdt_1m.csv"
OUTPUT_DIR            = Path("output")

# ────────────────────────── UTILITY HELPERS ──────────────────────────────── #


def _ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _random_slippage(price: float) -> float:
    """Return a random slippage amount in USDT for the given price."""
    pct = random.uniform(SLIPPAGE_MIN_PCT, SLIPPAGE_MAX_PCT)
    return price * pct


# ────────────────────────── DATA LOADING ─────────────────────────────────── #


def load_data(
    calendar_path: Path = CALENDAR_CSV,
    ohlcv_path: Path = OHLCV_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean both datasets.

    Returns
    -------
    events : DataFrame  – filtered High/Medium USD events with UTC datetime.
    ohlcv  : DataFrame  – BTC 1-min OHLCV with UTC datetime index.
    """

    # ── Economic Calendar ────────────────────────────────────────────────
    cal = pd.read_csv(calendar_path)

    # Normalise column names to lower-case for safety
    cal.columns = [c.strip().lower() for c in cal.columns]

    # Identify the timestamp column (try several common names)
    time_col = None
    for candidate in ("time", "date", "datetime", "timestamp", "date_time"):
        if candidate in cal.columns:
            time_col = candidate
            break
    if time_col is None:
        raise ValueError(
            f"Cannot find a time column in calendar CSV. Columns: {list(cal.columns)}"
        )

    cal["event_time"] = pd.to_datetime(cal[time_col], utc=True, errors="coerce")
    cal.dropna(subset=["event_time"], inplace=True)

    # Identify impact column
    impact_col = None
    for candidate in ("impact", "importance", "priority"):
        if candidate in cal.columns:
            impact_col = candidate
            break
    if impact_col is None:
        raise ValueError(
            f"Cannot find an impact column in calendar CSV. Columns: {list(cal.columns)}"
        )

    # Identify currency column
    currency_col = None
    for candidate in ("currency", "country", "ccy"):
        if candidate in cal.columns:
            currency_col = candidate
            break

    # Filter: only USD, High or Medium impact
    cal[impact_col] = cal[impact_col].astype(str).str.strip().str.lower()
    cal = cal[cal[impact_col].isin(["high", "medium"])]

    if currency_col is not None:
        cal[currency_col] = cal[currency_col].astype(str).str.strip().str.upper()
        cal = cal[cal[currency_col] == "USD"]

    events = (
        cal[["event_time"]]
        .drop_duplicates()
        .sort_values("event_time")
        .reset_index(drop=True)
    )

    # ── BTC 1-min OHLCV ─────────────────────────────────────────────────
    ohlcv = pd.read_csv(ohlcv_path)
    ohlcv.columns = [c.strip().lower() for c in ohlcv.columns]

    # Identify timestamp column in OHLCV
    ts_col = None
    for candidate in (
        "timestamp", "time", "date", "datetime", "open_time",
        "open time", "date_time", "close_time",
    ):
        if candidate in ohlcv.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(
            f"Cannot find a time column in OHLCV CSV. Columns: {list(ohlcv.columns)}"
        )

    # Handle both epoch-ms and ISO strings
    sample = ohlcv[ts_col].iloc[0]
    if isinstance(sample, (int, float, np.integer, np.floating)) or (
        isinstance(sample, str) and sample.replace(".", "").replace("-", "").isdigit()
    ):
        numeric_vals = pd.to_numeric(ohlcv[ts_col], errors="coerce")
        # If values > 1e12 they are in milliseconds
        if numeric_vals.median() > 1e12:
            ohlcv["datetime"] = pd.to_datetime(numeric_vals, unit="ms", utc=True)
        else:
            ohlcv["datetime"] = pd.to_datetime(numeric_vals, unit="s", utc=True)
    else:
        ohlcv["datetime"] = pd.to_datetime(ohlcv[ts_col], utc=True, errors="coerce")

    ohlcv.dropna(subset=["datetime"], inplace=True)

    # Ensure we have standard OHLCV columns
    rename_map = {}
    for target, candidates in {
        "open":   ["open", "o"],
        "high":   ["high", "h"],
        "low":    ["low", "l"],
        "close":  ["close", "c"],
        "volume": ["volume", "vol", "v"],
    }.items():
        for cand in candidates:
            if cand in ohlcv.columns and target not in ohlcv.columns:
                rename_map[cand] = target
                break
    ohlcv.rename(columns=rename_map, inplace=True)

    for required_col in ("open", "high", "low", "close"):
        if required_col not in ohlcv.columns:
            raise ValueError(
                f"OHLCV CSV missing required column '{required_col}'. "
                f"Available: {list(ohlcv.columns)}"
            )

    ohlcv = ohlcv[["datetime", "open", "high", "low", "close"]].copy()
    for col in ("open", "high", "low", "close"):
        ohlcv[col] = pd.to_numeric(ohlcv[col], errors="coerce")
    ohlcv.dropna(inplace=True)
    ohlcv.sort_values("datetime", inplace=True)
    ohlcv.set_index("datetime", inplace=True)

    print(f"[DATA] Loaded {len(events)} High/Medium USD events.")
    print(
        f"[DATA] Loaded {len(ohlcv)} OHLCV bars from "
        f"{ohlcv.index.min()} to {ohlcv.index.max()}."
    )

    return events, ohlcv


# ────────────────────────── EVENT DETECTION ──────────────────────────────── #


def find_upcoming_events(
    events: pd.DataFrame,
    ohlcv: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each event, compute the strangle-placement time (5 min before) and
    the current BTC price at that moment.

    Returns DataFrame with columns:
        event_time, placement_time, placement_price
    """
    ohlcv_start = ohlcv.index.min()
    ohlcv_end   = ohlcv.index.max()

    records: list = []
    for _, row in events.iterrows():
        evt = row["event_time"]
        placement = evt - pd.Timedelta(minutes=MINUTES_BEFORE_EVENT)

        # Skip if placement time is outside OHLCV range
        if placement < ohlcv_start or placement > ohlcv_end:
            continue

        # Find closest bar at or before placement time
        mask = ohlcv.index <= placement
        if not mask.any():
            continue
        bar = ohlcv.loc[mask].iloc[-1]
        records.append(
            {
                "event_time":      evt,
                "placement_time":  placement,
                "placement_price": float(bar["close"]),
            }
        )

    result = pd.DataFrame(records)
    print(f"[EVENTS] {len(result)} events matched with OHLCV data.")
    return result


# ────────────────────────── TRADE SIMULATION ─────────────────────────────── #


def simulate_trade(
    event_time: pd.Timestamp,
    placement_price: float,
    ohlcv: pd.DataFrame,
    equity: float,
) -> Optional[dict]:
    """
    Simulate a single strangle trade for one event.

    Returns a dict with trade details or None if no trigger occurred.
    """
    buy_stop  = placement_price + STRANGLE_OFFSET
    sell_stop = placement_price - STRANGLE_OFFSET

    # Position size: 1 % risk / SL distance
    risk_amount = equity * ACCOUNT_RISK_PCT
    position_size_btc = risk_amount / SL_DISTANCE   # BTC quantity

    # Time window: from event_time to event_time + MAX_HOLD
    window_start = event_time
    window_end   = event_time + pd.Timedelta(seconds=MAX_HOLD_SEC)

    bars = ohlcv.loc[
        (ohlcv.index >= window_start) & (ohlcv.index <= window_end)
    ]

    if bars.empty:
        return None

    # ── Phase 1: Find trigger ────────────────────────────────────────────
    trigger_time:  Optional[pd.Timestamp] = None
    trigger_price: Optional[float]        = None
    direction:     Optional[str]          = None   # "long" or "short"

    for bar_time, bar in bars.iterrows():
        if bar["high"] >= buy_stop:
            trigger_time  = bar_time
            trigger_price = buy_stop
            direction     = "long"
            break
        if bar["low"] <= sell_stop:
            trigger_time  = bar_time
            trigger_price = sell_stop
            direction     = "short"
            break

    if trigger_time is None:
        return None   # no trigger → no trade

    # Apply slippage at trigger
    slippage = _random_slippage(trigger_price)
    if direction == "long":
        entry_price = trigger_price + slippage
    else:
        entry_price = trigger_price - slippage

    # ── Phase 2: Confirmation filter (60 s after trigger) ────────────────
    confirm_deadline = trigger_time + pd.Timedelta(seconds=CONFIRMATION_WAIT_SEC)
    confirm_bars = ohlcv.loc[
        (ohlcv.index > trigger_time) & (ohlcv.index <= confirm_deadline)
    ]

    confirmed = False
    if not confirm_bars.empty:
        if direction == "long":
            max_move = confirm_bars["high"].max() - trigger_price
            confirmed = max_move >= CONFIRMATION_MIN
        else:
            max_move = trigger_price - confirm_bars["low"].min()
            confirmed = max_move >= CONFIRMATION_MIN

    if not confirmed:
        # Close at the price at confirmation deadline (or last available bar)
        if not confirm_bars.empty:
            exit_price = float(confirm_bars.iloc[-1]["close"])
        else:
            exit_price = entry_price   # no data → flat
        if direction == "long":
            pnl_usdt = (exit_price - entry_price) * position_size_btc
        else:
            pnl_usdt = (entry_price - exit_price) * position_size_btc

        return {
            "event_time":     event_time,
            "trigger_time":   trigger_time,
            "direction":      direction,
            "entry_price":    entry_price,
            "exit_price":     exit_price,
            "exit_reason":    "fakeout",
            "pnl_usdt":       pnl_usdt,
            "pnl_pct":        pnl_usdt / equity * 100,
            "pnl_r":          pnl_usdt / risk_amount,
            "slippage_usdt":  slippage,
            "confirmed":      False,
            "position_btc":   position_size_btc,
        }

    # ── Phase 3: Run trade with SL / TP / max-hold ──────────────────────
    if direction == "long":
        sl_price = entry_price - SL_DISTANCE
        tp_price = entry_price + TP_DISTANCE
    else:
        sl_price = entry_price + SL_DISTANCE
        tp_price = entry_price - TP_DISTANCE

    # Iterate bars after confirmation deadline up to max hold
    trade_bars = ohlcv.loc[
        (ohlcv.index > confirm_deadline) & (ohlcv.index <= window_end)
    ]

    exit_price  = None
    exit_reason = "max_hold"

    for bar_time, bar in trade_bars.iterrows():
        if direction == "long":
            # Check SL first (worst case)
            if bar["low"] <= sl_price:
                exit_price  = sl_price
                exit_reason = "sl"
                break
            # Check TP
            if bar["high"] >= tp_price:
                exit_price  = tp_price
                exit_reason = "tp"
                break
        else:   # short
            if bar["high"] >= sl_price:
                exit_price  = sl_price
                exit_reason = "sl"
                break
            if bar["low"] <= tp_price:
                exit_price  = tp_price
                exit_reason = "tp"
                break

    if exit_price is None:
        # Max hold reached – close at last available close
        if not trade_bars.empty:
            exit_price = float(trade_bars.iloc[-1]["close"])
        else:
            exit_price = entry_price
        exit_reason = "max_hold"

    if direction == "long":
        pnl_usdt = (exit_price - entry_price) * position_size_btc
    else:
        pnl_usdt = (entry_price - exit_price) * position_size_btc

    return {
        "event_time":     event_time,
        "trigger_time":   trigger_time,
        "direction":      direction,
        "entry_price":    entry_price,
        "exit_price":     exit_price,
        "exit_reason":    exit_reason,
        "pnl_usdt":       pnl_usdt,
        "pnl_pct":        pnl_usdt / equity * 100,
        "pnl_r":          pnl_usdt / risk_amount,
        "slippage_usdt":  slippage,
        "confirmed":      True,
        "position_btc":   position_size_btc,
    }


# ────────────────────────── STATISTICS ───────────────────────────────────── #


def compute_statistics(trades: pd.DataFrame, initial_equity: float) -> dict:
    """Compute all required backtest statistics from the trade log."""

    total   = len(trades)
    winners = trades[trades["pnl_usdt"] > 0]
    losers  = trades[trades["pnl_usdt"] <= 0]

    wins   = len(winners)
    losses = len(losers)

    winrate = wins / total * 100 if total > 0 else 0.0

    gross_profit = winners["pnl_usdt"].sum() if wins > 0 else 0.0
    gross_loss   = abs(losers["pnl_usdt"].sum()) if losses > 0 else 0.0
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    avg_win  = winners["pnl_r"].mean() if wins > 0 else 0.0
    avg_loss = losers["pnl_r"].mean() if losses > 0 else 0.0

    expectancy_r = trades["pnl_r"].mean() if total > 0 else 0.0

    # Equity curve (compounded 1 % risk)
    equity_curve = [initial_equity]
    eq = initial_equity
    for _, t in trades.iterrows():
        eq += t["pnl_usdt"]
        equity_curve.append(eq)
    equity_curve = np.array(equity_curve)

    # Max drawdown
    running_max = np.maximum.accumulate(equity_curve)
    drawdown    = (running_max - equity_curve) / running_max * 100
    max_dd_pct  = drawdown.max()

    # CAGR
    final_eq = equity_curve[-1]
    if total > 0:
        first_time = trades["event_time"].min()
        last_time  = trades["event_time"].max()
        years = max((last_time - first_time).total_seconds() / (365.25 * 86400), 1e-6)
    else:
        years = 1.0
    cagr = ((final_eq / initial_equity) ** (1 / years) - 1) * 100

    # Sharpe Ratio (daily returns, risk-free = 0)
    if total > 1:
        daily_returns = trades.set_index("event_time")["pnl_pct"].resample("D").sum()
        daily_returns = daily_returns[daily_returns != 0]
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return {
        "total_trades":   total,
        "wins":           wins,
        "losses":         losses,
        "winrate":        winrate,
        "profit_factor":  profit_factor,
        "expectancy_r":   expectancy_r,
        "avg_win_r":      avg_win,
        "avg_loss_r":     avg_loss,
        "max_dd_pct":     max_dd_pct,
        "cagr_pct":       cagr,
        "sharpe":         sharpe,
        "final_equity":   final_eq,
        "equity_curve":   equity_curve,
    }


# ────────────────────────── REPORTING & PLOTS ────────────────────────────── #


def _format_stats(stats: dict) -> str:
    """Return a human-readable statistics report string."""

    lines = [
        "=" * 60,
        "  AutoNews Strangle Backtest – Statistics Report",
        "=" * 60,
        f"  Total Trades     : {stats['total_trades']}",
        f"  Winners          : {stats['wins']}",
        f"  Losers           : {stats['losses']}",
        f"  Winrate          : {stats['winrate']:.2f} %",
        f"  Profit Factor    : {stats['profit_factor']:.2f}",
        f"  Expectancy (R)   : {stats['expectancy_r']:.2f}",
        f"  Avg Win (R)      : {stats['avg_win_r']:.2f}",
        f"  Avg Loss (R)     : {stats['avg_loss_r']:.2f}",
        f"  Max Drawdown     : {stats['max_dd_pct']:.2f} %",
        f"  CAGR             : {stats['cagr_pct']:.2f} %",
        f"  Sharpe Ratio     : {stats['sharpe']:.2f}",
        f"  Final Equity     : {stats['final_equity']:.2f} USDT",
        "-" * 60,
    ]

    # PASS / FAIL verdict
    passed = (
        stats["expectancy_r"] > PASS_EXPECTANCY_MIN
        and stats["max_dd_pct"] < PASS_MAX_DD_PCT
        and stats["profit_factor"] > PASS_PF_MIN
    )
    if passed:
        lines.append("  ✅  Backtest PASSED")
        lines.append(
            f"      Expectancy {stats['expectancy_r']:.2f}R > {PASS_EXPECTANCY_MIN}R, "
            f"Max DD {stats['max_dd_pct']:.2f}% < {PASS_MAX_DD_PCT}%, "
            f"PF {stats['profit_factor']:.2f} > {PASS_PF_MIN}"
        )
    else:
        lines.append("  ❌  Backtest FAILED")
        reasons = []
        if stats["expectancy_r"] <= PASS_EXPECTANCY_MIN:
            reasons.append(
                f"Expectancy {stats['expectancy_r']:.2f}R <= {PASS_EXPECTANCY_MIN}R"
            )
        if stats["max_dd_pct"] >= PASS_MAX_DD_PCT:
            reasons.append(
                f"Max DD {stats['max_dd_pct']:.2f}% >= {PASS_MAX_DD_PCT}%"
            )
        if stats["profit_factor"] <= PASS_PF_MIN:
            reasons.append(
                f"PF {stats['profit_factor']:.2f} <= {PASS_PF_MIN}"
            )
        lines.append("      Reason(s): " + " | ".join(reasons))

    lines.append("=" * 60)
    return "\n".join(lines)


def save_equity_curve(stats: dict, output_dir: Path = OUTPUT_DIR) -> str:
    """Plot and save the equity curve as PNG."""
    _ensure_dir(output_dir)
    filepath = output_dir / "equity_curve.png"

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(stats["equity_curve"], linewidth=1.2, color="#2196F3")
    ax.set_title("AutoNews Strangle Backtest 2023-2026", fontsize=14, weight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (USDT)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Equity curve saved → {filepath}")
    return str(filepath)


def save_scatter_plot(trades: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> str:
    """Plot winners vs losers scatter and save as PNG."""
    _ensure_dir(output_dir)
    filepath = output_dir / "scatter_win_loss.png"

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in trades["pnl_r"]]
    ax.scatter(range(len(trades)), trades["pnl_r"], c=colors, alpha=0.7, s=30)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Winners vs Losers (R-Multiple)", fontsize=14, weight="bold")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("P&L (R)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Scatter plot saved → {filepath}")
    return str(filepath)


def save_trade_log(trades: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> str:
    """Save full trade log as CSV."""
    _ensure_dir(output_dir)
    filepath = output_dir / "trade_log.csv"
    trades.to_csv(filepath, index=False)
    print(f"[LOG]  Trade log saved → {filepath}")
    return str(filepath)


def save_report(report_text: str, output_dir: Path = OUTPUT_DIR) -> str:
    """Save statistics report as TXT."""
    _ensure_dir(output_dir)
    filepath = output_dir / "backtest_report.txt"
    filepath.write_text(report_text, encoding="utf-8")
    print(f"[RPT]  Report saved   → {filepath}")
    return str(filepath)


# ────────────────────────── MAIN BACKTEST RUNNER ─────────────────────────── #


def run_full_backtest(
    calendar_path: Path = CALENDAR_CSV,
    ohlcv_path: Path = OHLCV_CSV,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    Execute the complete backtest pipeline:
      1. Load data
      2. Find matching events
      3. Simulate all trades
      4. Compute statistics
      5. Generate plots & reports

    Returns the statistics dictionary.
    """
    random.seed(RANDOM_SEED)

    print("\n" + "=" * 60)
    print("  AutoNews Strangle Backtester – Phase 1")
    print("=" * 60 + "\n")

    # 1. Load data
    events, ohlcv = load_data(calendar_path, ohlcv_path)

    # 2. Find events with placement prices
    event_df = find_upcoming_events(events, ohlcv)

    if event_df.empty:
        print("[WARN] No events found in OHLCV range. Aborting.")
        return {}

    # 3. Simulate trades
    equity = INITIAL_EQUITY
    trade_records: List[dict] = []

    for _, ev in event_df.iterrows():
        result = simulate_trade(
            event_time=ev["event_time"],
            placement_price=ev["placement_price"],
            ohlcv=ohlcv,
            equity=equity,
        )
        if result is not None:
            trade_records.append(result)
            # Update equity (compounded)
            equity += result["pnl_usdt"]
            # Prevent negative equity
            if equity <= 0:
                print("[STOP] Account blown. Stopping backtest.")
                break

    if not trade_records:
        print("[WARN] No trades triggered. Aborting.")
        return {}

    trades = pd.DataFrame(trade_records)

    # 4. Compute statistics
    stats = compute_statistics(trades, INITIAL_EQUITY)

    # 5. Report
    report_text = _format_stats(stats)
    print("\n" + report_text + "\n")

    # 6. Save outputs
    save_trade_log(trades, output_dir)
    save_equity_curve(stats, output_dir)
    save_scatter_plot(trades, output_dir)
    save_report(report_text, output_dir)

    return stats


# ─────────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    run_full_backtest()
