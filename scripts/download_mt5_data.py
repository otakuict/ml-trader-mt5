from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_D1
START_DATE = datetime(2001, 1, 1)
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "gold-data.csv"


def download_data() -> bool:
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return False
    try:
        end_date = datetime.now()
        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, START_DATE, end_date)
        if rates is None or len(rates) == 0:
            print("No data returned. Check symbol name and history in MT5.")
            return False
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_PATH, index=False)
        print(f"Saved {len(df)} rows to {OUT_PATH}")
        return True
    finally:
        mt5.shutdown()


def main() -> int:
    return 0 if download_data() else 1


if __name__ == "__main__":
    raise SystemExit(main())
