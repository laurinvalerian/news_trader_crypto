"""
tests/test_backtester.py
=========================
Unit tests for news_strangle_backtester.py

Tests cover:
  - Data loading and column detection
  - Event filtering (only High/Medium USD)
  - Trade simulation logic (trigger, confirmation, SL/TP, fakeout, max-hold)
  - Statistics computation
  - Report formatting and PASS/FAIL verdict
"""

import datetime as dt
import sys
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import news_strangle_backtester as bt


# ─────────────────────────── FIXTURES ────────────────────────────────────── #


@pytest.fixture
def sample_ohlcv():
    """Create a small synthetic 1-min OHLCV DataFrame for testing."""
    base_time = pd.Timestamp("2023-06-01 14:00:00", tz="UTC")
    n = 300  # 5 hours of data
    times = pd.date_range(start=base_time, periods=n, freq="min", tz="UTC")
    base_price = 27000.0

    # Simple upward drift for testing
    closes = base_price + np.arange(n) * 5.0
    df = pd.DataFrame({
        "open":  closes - 2.0,
        "high":  closes + 10.0,
        "low":   closes - 10.0,
        "close": closes,
    }, index=times)
    df.index.name = "datetime"
    return df


@pytest.fixture
def sample_events():
    """Create sample events DataFrame."""
    return pd.DataFrame({
        "event_time": [
            pd.Timestamp("2023-06-01 14:30:00", tz="UTC"),
            pd.Timestamp("2023-06-01 15:30:00", tz="UTC"),
        ]
    })


@pytest.fixture
def sample_calendar_csv(tmp_path):
    """Create a temporary economic calendar CSV."""
    data = {
        "time": [
            "2023-06-01 14:30:00",
            "2023-06-01 15:30:00",
            "2023-06-01 16:00:00",
            "2023-06-01 17:00:00",
        ],
        "currency": ["USD", "USD", "EUR", "USD"],
        "impact": ["High", "Medium", "High", "Low"],
        "event": ["NFP", "CPI", "ECB Rate", "Minor Data"],
    }
    path = tmp_path / "calendar.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


@pytest.fixture
def sample_ohlcv_csv(tmp_path):
    """Create a temporary OHLCV CSV."""
    base_time = pd.Timestamp("2023-06-01 14:00:00", tz="UTC")
    n = 300
    times = pd.date_range(start=base_time, periods=n, freq="min", tz="UTC")
    base_price = 27000.0
    closes = base_price + np.arange(n) * 5.0

    df = pd.DataFrame({
        "timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
        "open":  closes - 2.0,
        "high":  closes + 10.0,
        "low":   closes - 10.0,
        "close": closes,
        "volume": np.random.uniform(1, 100, n),
    })
    path = tmp_path / "ohlcv.csv"
    df.to_csv(path, index=False)
    return path


# ─────────────────────────── TESTS: load_data ────────────────────────────── #


class TestLoadData:
    """Tests for the load_data function."""

    def test_loads_and_filters_correctly(self, sample_calendar_csv, sample_ohlcv_csv):
        """Should load data and filter only USD High/Medium events."""
        events, ohlcv = bt.load_data(sample_calendar_csv, sample_ohlcv_csv)

        # Should have 2 events (USD High + USD Medium), not EUR or Low
        assert len(events) == 2
        assert len(ohlcv) == 300

    def test_ohlcv_has_required_columns(self, sample_calendar_csv, sample_ohlcv_csv):
        """OHLCV should have open, high, low, close columns."""
        _, ohlcv = bt.load_data(sample_calendar_csv, sample_ohlcv_csv)
        for col in ("open", "high", "low", "close"):
            assert col in ohlcv.columns

    def test_missing_time_column_raises(self, tmp_path, sample_ohlcv_csv):
        """Should raise ValueError when calendar has no recognizable time column."""
        bad_csv = tmp_path / "bad_cal.csv"
        pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="Cannot find a time column"):
            bt.load_data(bad_csv, sample_ohlcv_csv)


# ─────────────────────── TESTS: find_upcoming_events ─────────────────────── #


class TestFindUpcomingEvents:
    """Tests for event matching with OHLCV data."""

    def test_returns_correct_events(self, sample_events, sample_ohlcv):
        """Should match events that fall within OHLCV range."""
        result = bt.find_upcoming_events(sample_events, sample_ohlcv)
        # Both events should be within the 5-hour OHLCV window
        assert len(result) == 2
        assert "placement_price" in result.columns
        assert "placement_time" in result.columns

    def test_out_of_range_events_skipped(self, sample_ohlcv):
        """Events outside OHLCV time range should be skipped."""
        far_events = pd.DataFrame({
            "event_time": [pd.Timestamp("2020-01-01 12:00:00", tz="UTC")]
        })
        result = bt.find_upcoming_events(far_events, sample_ohlcv)
        assert len(result) == 0


# ──────────────────────── TESTS: simulate_trade ──────────────────────────── #


class TestSimulateTrade:
    """Tests for single trade simulation."""

    def test_no_trigger_returns_none(self):
        """If price never reaches buy/sell stop, should return None."""
        # Create flat price data that won't trigger any stop
        base_time = pd.Timestamp("2023-06-01 14:30:00", tz="UTC")
        times = pd.date_range(start=base_time, periods=120, freq="min", tz="UTC")
        price = 27000.0  # placement is also 27000

        ohlcv = pd.DataFrame({
            "open":  [price] * 120,
            "high":  [price + 50] * 120,   # only +50, never reaches +400
            "low":   [price - 50] * 120,   # only -50, never reaches -400
            "close": [price] * 120,
        }, index=times)

        result = bt.simulate_trade(
            event_time=base_time,
            placement_price=price,
            ohlcv=ohlcv,
            equity=10000.0,
        )
        assert result is None

    def test_long_trigger_with_tp(self):
        """Simulate a long trigger that hits TP."""
        import random
        random.seed(42)

        base_time = pd.Timestamp("2023-06-01 14:30:00", tz="UTC")
        n = 180  # 3 hours
        times = pd.date_range(start=base_time, periods=n, freq="min", tz="UTC")
        placement_price = 27000.0

        # Create price that spikes sharply to trigger buy stop AND pass the
        # 60-second confirmation filter (+300 USDT within 1 minute bar).
        # Then continues rising to hit TP.
        prices = np.full(n, placement_price)
        # Minute 0: spike triggers buy stop at 27400
        prices[0] = placement_price + 500
        # Minute 1: price jumps further to confirm (+300 above trigger)
        prices[1] = placement_price + 800
        # Remaining minutes: price continues rising to hit TP (entry + 6000)
        for i in range(2, n):
            prices[i] = placement_price + 800 + (i - 2) * 60

        ohlcv = pd.DataFrame({
            "open":  prices - 5,
            "high":  prices + 50,
            "low":   prices - 50,
            "close": prices,
        }, index=times)

        result = bt.simulate_trade(
            event_time=base_time,
            placement_price=placement_price,
            ohlcv=ohlcv,
            equity=10000.0,
        )

        assert result is not None
        assert result["direction"] == "long"
        assert result["confirmed"] is True
        assert result["pnl_usdt"] > 0

    def test_fakeout_detection(self):
        """Trade should close as fakeout if confirmation fails."""
        import random
        random.seed(42)

        base_time = pd.Timestamp("2023-06-01 14:30:00", tz="UTC")
        n = 180
        times = pd.date_range(start=base_time, periods=n, freq="min", tz="UTC")
        placement_price = 27000.0

        # Price spikes up to trigger buy stop, then stays flat (no +300 confirmation)
        prices = np.full(n, placement_price)
        # Trigger on first bar
        prices[0] = placement_price + 450   # triggers buy stop
        # After trigger, price stays near trigger level (< +300 move)
        prices[1:5] = placement_price + 410

        ohlcv = pd.DataFrame({
            "open":  prices - 5,
            "high":  prices + 10,
            "low":   prices - 10,
            "close": prices,
        }, index=times)

        result = bt.simulate_trade(
            event_time=base_time,
            placement_price=placement_price,
            ohlcv=ohlcv,
            equity=10000.0,
        )

        assert result is not None
        assert result["exit_reason"] == "fakeout"
        assert result["confirmed"] is False


# ──────────────────────── TESTS: compute_statistics ──────────────────────── #


class TestComputeStatistics:
    """Tests for statistics computation."""

    def test_all_winners(self):
        """Stats should reflect 100% winrate when all trades win."""
        trades = pd.DataFrame({
            "event_time": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"],
                                         utc=True),
            "pnl_usdt": [100.0, 200.0, 150.0],
            "pnl_r":    [1.0, 2.0, 1.5],
            "pnl_pct":  [1.0, 2.0, 1.5],
        })
        stats = bt.compute_statistics(trades, 10000.0)

        assert stats["total_trades"] == 3
        assert stats["winrate"] == 100.0
        assert stats["wins"] == 3
        assert stats["losses"] == 0
        assert stats["profit_factor"] == float("inf")
        assert stats["expectancy_r"] == pytest.approx(1.5, rel=0.01)

    def test_mixed_trades(self):
        """Stats should correctly compute with mixed wins and losses."""
        trades = pd.DataFrame({
            "event_time": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"],
                utc=True,
            ),
            "pnl_usdt": [600.0, -100.0, 600.0, -100.0],
            "pnl_r":    [6.0, -1.0, 6.0, -1.0],
            "pnl_pct":  [6.0, -1.0, 6.0, -1.0],
        })
        stats = bt.compute_statistics(trades, 10000.0)

        assert stats["total_trades"] == 4
        assert stats["winrate"] == 50.0
        assert stats["profit_factor"] == pytest.approx(6.0, rel=0.01)
        assert stats["expectancy_r"] == pytest.approx(2.5, rel=0.01)

    def test_max_drawdown(self):
        """Max drawdown should be computed correctly."""
        trades = pd.DataFrame({
            "event_time": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-03-01"],
                utc=True,
            ),
            "pnl_usdt": [-500.0, -300.0, 1000.0],
            "pnl_r":    [-5.0, -3.0, 10.0],
            "pnl_pct":  [-5.0, -3.0, 10.0],
        })
        stats = bt.compute_statistics(trades, 10000.0)
        # Equity: 10000 → 9500 → 9200 → 10200
        # Max DD = (10000 - 9200) / 10000 = 8%
        assert stats["max_dd_pct"] == pytest.approx(8.0, rel=0.01)


# ──────────────────────── TESTS: reporting ───────────────────────────────── #


class TestReporting:
    """Tests for report formatting and verdict."""

    def test_passed_verdict(self):
        """Report should say PASSED when all criteria are met."""
        stats = {
            "total_trades": 100,
            "wins": 30,
            "losses": 70,
            "winrate": 30.0,
            "profit_factor": 2.5,
            "expectancy_r": 2.0,
            "avg_win_r": 7.5,
            "avg_loss_r": -1.0,
            "max_dd_pct": 10.0,
            "cagr_pct": 50.0,
            "sharpe": 1.5,
            "final_equity": 15000.0,
            "equity_curve": np.array([10000.0, 15000.0]),
        }
        report = bt._format_stats(stats)
        assert "PASSED" in report

    def test_failed_verdict(self):
        """Report should say FAILED when criteria are not met."""
        stats = {
            "total_trades": 100,
            "wins": 30,
            "losses": 70,
            "winrate": 30.0,
            "profit_factor": 1.2,       # below 1.8
            "expectancy_r": 0.5,        # below 1.5
            "avg_win_r": 3.0,
            "avg_loss_r": -1.0,
            "max_dd_pct": 20.0,         # above 15%
            "cagr_pct": 10.0,
            "sharpe": 0.5,
            "final_equity": 11000.0,
            "equity_curve": np.array([10000.0, 11000.0]),
        }
        report = bt._format_stats(stats)
        assert "FAILED" in report


# ──────────────────────── TESTS: helper functions ────────────────────────── #


class TestHelpers:
    """Tests for utility helper functions."""

    def test_random_slippage_range(self):
        """Slippage should be within the configured range."""
        import random
        random.seed(42)

        price = 30000.0
        for _ in range(100):
            slip = bt._random_slippage(price)
            assert slip >= price * bt.SLIPPAGE_MIN_PCT
            assert slip <= price * bt.SLIPPAGE_MAX_PCT

    def test_ensure_dir(self, tmp_path):
        """_ensure_dir should create the directory."""
        new_dir = tmp_path / "test_output" / "nested"
        bt._ensure_dir(new_dir)
        assert new_dir.exists()


# ──────────────────────── TESTS: plot saving ─────────────────────────────── #


class TestPlotSaving:
    """Tests for plot generation (verify files are created)."""

    def test_equity_curve_saved(self, tmp_path):
        """Equity curve PNG should be created."""
        stats = {
            "equity_curve": np.array([10000, 10100, 10050, 10200, 10300]),
        }
        path = bt.save_equity_curve(stats, tmp_path)
        assert Path(path).exists()
        assert path.endswith(".png")

    def test_scatter_plot_saved(self, tmp_path):
        """Scatter plot PNG should be created."""
        trades = pd.DataFrame({
            "pnl_r": [1.5, -1.0, 7.5, -1.0, 3.0],
        })
        path = bt.save_scatter_plot(trades, tmp_path)
        assert Path(path).exists()
        assert path.endswith(".png")

    def test_trade_log_saved(self, tmp_path):
        """Trade log CSV should be created."""
        trades = pd.DataFrame({
            "event_time": ["2023-01-01"],
            "direction": ["long"],
            "pnl_usdt": [100.0],
        })
        path = bt.save_trade_log(trades, tmp_path)
        assert Path(path).exists()
        assert path.endswith(".csv")

    def test_report_saved(self, tmp_path):
        """Report TXT should be created."""
        path = bt.save_report("Test report content", tmp_path)
        assert Path(path).exists()
        assert Path(path).read_text() == "Test report content"
