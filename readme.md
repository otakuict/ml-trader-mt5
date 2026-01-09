# ml-trader - XM MT5 Demo ML Bot (GOLD)

A simple **demo** trading project for **XM MetaTrader 5 (MT5)** using a **machine-learning signal**.
Default timeframe: **H4** (model trains on daily bars derived from H4).

> Use **demo** for learning/testing. Real-money trading is risky and often age-restricted.

---

## Project layout

ml-trader/
- requirements.txt
- run_bot.bat
- retrain_all.bat
- readme_daytrade.md
- data/
- logs/
- models/
- scripts/
  - data_loader.py
  - download_mt5_data.py
  - backtest_bt.py
  - backtest_ml.py
  - train_model.py
  - auto_retrain.py
  - predict_signal.py
  - run_forever.py
  - test_order.py

---

## Requirements

- Windows PC (recommended)
- XM MT5 installed + logged into a demo account
- In MT5 Market Watch, the symbol name is **GOLD**
- Python 3.9+ (3.10+ recommended)

---

## Setup (MT5)

1. Open MT5 and login to an **XM demo** account
2. Market Watch (Ctrl+M) + right click + Show All
3. Confirm you see **GOLD**
4. Open a **GOLD** chart and set timeframe to **H4**
5. Scroll back to load more history (important)

If you already have a CSV, place it in `data/` and use it directly (see Data files).

---

## Setup (Python)

1) Create + activate venv:
- python -m venv .venv
- .\.venv\Scripts\activate

2) Install packages:
- pip install -r requirements.txt

---

## Data files

Default data file (used by all scripts):
- `data/gold-data-h4.csv`

`download_mt5_data.py` uses a date range (2000 -> now) and **overwrites** the file.
If you want to keep a long historical CSV, do not overwrite it, or change the output path.

---

## Run order (first time)

1) Download GOLD H4 candles to CSV:
- python scripts/download_mt5_data.py

2) Backtest starter strategy (SMA cross):
- python scripts/backtest_bt.py

2b) Backtest ML strategy (uses saved model):
- python scripts/backtest_ml.py

3) Train ML classification model (incremental updates):
- python scripts/train_model.py

4) Check prediction + signal:
- python scripts/predict_signal.py

5) Run the bot loop:
- python scripts/run_forever.py

---

## Retrain flow

Manual retrain (refresh data + train + backtest + predict):
- .\retrain_all.bat

Auto-retrain (only replace if performance improves):
- python scripts/auto_retrain.py

---

## Logs and models

- Training logs: `logs/train_YYYYMMDD_HHMMSS.log`
- Auto-retrain logs: `logs/auto_retrain_YYYYMMDD_HHMMSS.log`
- Timestamped models: `models/model_YYYYMMDD_HHMMSS.joblib`
- Latest model copy: `models/model.joblib`
- Model metadata: `models/model_meta.json`

`run_forever.py` prints which model it is using at runtime.

---

## Run-forever behavior

`run_forever.py` checks every 5 minutes and:
- Builds a daily signal from H4 data
- Places one trade per day at **23:55 local time** (configurable)
- Uses ATR-based SL/TP if enabled

Safety switches (top of `scripts/run_forever.py`):
- `DRY_RUN` - no orders if True
- `ALLOW_REAL` - must be True to trade a non-demo account
- `LOT_SIZE` - small default size for demo testing
- `ML_TRADING_ENABLED` - set False to pause ML trading
- `CLASS_BUY_THRESHOLD` / `CLASS_SELL_THRESHOLD` - confidence thresholds (default 0.85/0.15)
- `USE_ATR_SLTP`, `SL_ATR_MULT`, `TP_ATR_MULT` - ATR-based stop/take-profit (default SLx=1.2 TPx=4.0)
- `TREND_FILTER` - only buy above MA50, sell below MA50
- `TREND_MA_PERIOD` - moving average length used by the trend filter (default 50)
- `DAILY_TRADE_ONLY` - trade once per day at the configured time
- `DAILY_TRADE_HOUR` / `DAILY_TRADE_MINUTE` - local time for daily entry (default 23:55)
- `DAILY_TRADE_DIR_THRESHOLD` - probability cutoff for daily direction (>= BUY else SELL)
- `DAILY_CLOSE_EXISTING` - close existing position before the daily trade
- `DAILY_CONF_BUY` / `DAILY_CONF_SELL` - confidence band for model-driven direction (default 0.65/0.35)
- `DAILY_FALLBACK_TREND` - use MA50 trend when probability is between the band
- `DAILY_USE_MODEL` - set False to use pure MA50 trend for daily direction
- `FORCE_DAILY_TRADE` - force at least one trade per day
- `DAILY_TRADE_HOUR` / `DAILY_TRADE_MINUTE` - time for the daily forced trade
- `DAILY_TRADE_DIR_THRESHOLD` - probability threshold to pick BUY vs SELL
- `DAILY_TRADE_IGNORE_TREND` - ignore MA50 filter for forced daily trade

---

## One-time test order

Use `scripts/test_order.py` to place and close a tiny demo order and print recent history:
- python scripts/test_order.py

---

## ML backtest notes

`backtest_ml.py` supports:
- Walk-forward evaluation (train window + test window)
- Confidence thresholds for trades
- ATR-based stop-loss / take-profit
- Trend filter using MA50
- Daily forced trade (same settings as live)
Default walk-forward windows: 1000 train / 1 test.

`train_model.py` uses incremental learning (updates the previous model if new data exists).

---

## Day trade (M30)

See `readme_daytrade.md` for M30 setup.
