import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_D1
BARS = 2000
SLEEP_SECONDS = 300
DRY_RUN = True

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold_d1.csv"


def download_data() -> bool:
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return False
    try:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
        if rates is None or len(rates) == 0:
            print("No data returned. Check symbol name and history in MT5.")
            return False
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return True
    finally:
        mt5.shutdown()


def latest_bar_time() -> Optional[datetime]:
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        return None
    return pd.to_datetime(df.iloc[-1]["time"]).to_pydatetime()


def main() -> int:
    last_seen = None
    print("Running forever. Press Ctrl+C to stop.")
    while True:
        if download_data():
            current = latest_bar_time()
            if current and current != last_seen:
                last_seen = current
                print(f"New D1 candle detected at {current}")
                if DRY_RUN:
                    print("DRY_RUN is on. No orders will be sent.")
                else:
                    print("Trading logic would run here.")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
