# News Trader Crypto – AutoNews Strangle Backtester

Phase 1: Standalone backtester for the **"Free Crypto News Strangle Strategy"** on BTCUSDT Perpetual Futures (Binance).

## Strategy Overview

- **Instrument**: BTCUSDT Perpetual Futures
- **Period**: January 2023 – March 2026
- **Events**: Only High + Medium impact USD economic events
- **Entry**: Strangle placed 5 minutes before event (Buy-Stop +400, Sell-Stop −400 from current price)
- **Risk**: 1% of account equity per trade
- **SL**: 800 USDT (fixed) | **TP**: 6,000 USDT (fixed) → 1:7.5 RR
- **Confirmation**: 60-second filter after trigger (+300 USDT minimum move required)
- **Max Hold**: 2 hours, then auto-close
- **Slippage**: Random +0.15% to +0.50% at news release

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

Download the following datasets and place the CSV files in a `data/` directory:

1. **Economic Calendar** (CSV with columns: `time`, `impact`, `currency`):
   - Source: https://www.kaggle.com/datasets/youneseloiarm/global-economic-calendar
   - Save as: `data/economic_calendar.csv`

2. **BTC 1-Minute OHLCV** (CSV with columns: `timestamp`, `open`, `high`, `low`, `close`):
   - Source: https://www.kaggle.com/datasets/sid17a/btcusdt-1min-perpetual-dataset-binance-apis
   - Or: https://data.binance.vision (1m klines)
   - Or: https://www.kaggle.com/datasets/youneseloiarm/bitcoin-btcusdt
   - Save as: `data/btcusdt_1m.csv`

### Using Synthetic Data (for testing)

```bash
python generate_sample_data.py
```

This generates synthetic sample data in `data/` for testing the backtester logic.

## Running the Backtester

```bash
python news_strangle_backtester.py
```

### Output

The backtester produces:
- `output/trade_log.csv` – Complete trade log (entry/exit times, prices, P&L, slippage)
- `output/backtest_report.txt` – Statistics report with PASS/FAIL verdict
- `output/equity_curve.png` – Equity curve plot
- `output/scatter_win_loss.png` – Winners vs losers scatter plot

### PASS/FAIL Criteria

The backtest is marked **PASSED** only if ALL of these conditions are met:
- Expectancy > +1.5R
- Max Drawdown < 15%
- Profit Factor > 1.8

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Project Structure

```
news_trader_crypto/
├── news_strangle_backtester.py   # Main backtester (single-file, standalone)
├── generate_sample_data.py       # Synthetic data generator for testing
├── requirements.txt              # Python dependencies
├── tests/
│   ├── __init__.py
│   └── test_backtester.py        # Unit tests (19 tests)
├── data/                         # User-provided CSV data (gitignored)
│   ├── economic_calendar.csv
│   └── btcusdt_1m.csv
└── output/                       # Generated outputs (gitignored)
    ├── trade_log.csv
    ├── backtest_report.txt
    ├── equity_curve.png
    └── scatter_win_loss.png
```

## Configuration

All strategy parameters are defined as constants at the top of `news_strangle_backtester.py` and are easily adjustable:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `STRANGLE_OFFSET` | 400 USDT | Distance from current price for buy/sell stops |
| `SL_DISTANCE` | 800 USDT | Fixed stop-loss distance |
| `TP_DISTANCE` | 6,000 USDT | Fixed take-profit distance (1:7.5 RR) |
| `ACCOUNT_RISK_PCT` | 0.01 (1%) | Equity risked per trade |
| `INITIAL_EQUITY` | 10,000 USDT | Starting account balance |
| `CONFIRMATION_WAIT_SEC` | 60 | Seconds to wait for confirmation |
| `CONFIRMATION_MIN` | 300 USDT | Minimum move to confirm trade |
| `MAX_HOLD_SEC` | 7,200 (2h) | Maximum trade hold time |
| `SLIPPAGE_MIN_PCT` | 0.15% | Minimum slippage on news candles |
| `SLIPPAGE_MAX_PCT` | 0.50% | Maximum slippage on news candles |
