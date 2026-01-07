# ml-trader — XM MT5 Demo ML Bot (GOLD, D1)

A simple **demo** trading project for **XM MetaTrader 5 (MT5)** that trades **GOLD** on **Daily (D1)** using a **machine-learning regression** signal.

It includes:
- MT5 → CSV data download
- Backtesting with Backtrader
- ML regression training + prediction
- A “run forever” bot loop (terminal stays open)
- Windows Startup launch (no Task Scheduler)

> Use **demo** for learning/testing. Real-money trading is risky and often age-restricted.

---

## Project layout

ml-trader/
- requirements.txt
- run_bot.bat
- data/
- logs/
- models/
- scripts/
  - download_mt5_data.py
  - backtest_bt.py
  - train_model.py
  - predict_signal.py
  - run_forever.py

---

## Requirements

- Windows PC (recommended)
- XM MT5 installed + logged into a demo account
- In MT5 Market Watch, the symbol name is **GOLD**
- Python 3.9+ (3.10+ recommended)

---

## Setup (MT5)

1. Open MT5 and login to an **XM demo** account
2. Market Watch (Ctrl+M) → right click → Show All
3. Confirm you see **GOLD**
4. Open a **GOLD** chart → set timeframe to **D1**
5. Scroll back to load more history (important)
6. Optional: If you already have a CSV (example: data/GOLD_Daily_200106040000_202601070000.csv), place it in data/ and skip manual scrolling

---

## Setup (Python)

1) Create + activate venv:
- python -m venv .venv
- .\.venv\Scripts\activate

2) Install packages:
- pip install -r requirements.txt

requirements.txt should include:
- MetaTrader5
- pandas
- backtrader
- scikit-learn
- joblib

---

## Run order (first time)

1) Download GOLD D1 candles to CSV:
- python scripts/download_mt5_data.py

2) Backtest starter strategy:
- python scripts/backtest_bt.py

3) Train ML regression model:
- python scripts/train_model.py

4) Check prediction + signal:
- python scripts/predict_signal.py

5) Run the bot forever (keeps terminal open):
- python scripts/run_forever.py

---

## Auto-run on Windows startup (one terminal window)

1) Create run_bot.bat in the project root:

@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python scripts\run_forever.py
pause

2) Add it to Startup:
- Win + R → shell:startup
- Put a shortcut to run_bot.bat in that folder

Now the bot starts when you log in and keeps a terminal window open.

---

## Notes

- If MT5 returns no data, open GOLD D1 and scroll left to load history.
- If your XM symbol isn’t exactly GOLD, change SYMBOL="GOLD" inside scripts.
- For stability, set your PC Sleep/Hibernate to “Never” (or long enough).

---

## Suggested safety defaults (demo)
- Small lot size (e.g., 0.01)
- One position at a time
- Trade at most once per new D1 candle
- Use an ATR-based threshold so it doesn’t trade tiny/noisy predictions
