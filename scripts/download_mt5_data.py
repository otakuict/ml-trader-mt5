from datetime import datetime, timedelta

import MetaTrader5 as mt5
import pandas as pd

from data_loader import load_price_data
from settings import DATA_PATH, START_DATE, SYMBOL

TIMEFRAME = mt5.TIMEFRAME_H4
OUT_PATH = DATA_PATH
RESYNC_FULL_RANGE = False


def download_data() -> bool:
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return False
    try:
        end_date = datetime.now()
        existing = None
        existing_count = 0
        if OUT_PATH.exists():
            existing = load_price_data(OUT_PATH)
            existing = existing.dropna(subset=["time"]).sort_values("time")
            if not existing.empty:
                existing_count = len(existing)
                if RESYNC_FULL_RANGE:
                    start_date = START_DATE
                else:
                    last_time = existing["time"].iloc[-1].to_pydatetime()
                    start_date = last_time + timedelta(seconds=1)
            else:
                start_date = START_DATE
        else:
            start_date = START_DATE

        rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)
        if rates is None or len(rates) == 0:
            if existing is not None:
                print("No new data returned. Keeping existing file.")
                return True
            print("No data returned. Check symbol name and history in MT5.")
            return False

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df], ignore_index=True, sort=False)
            df = df.drop_duplicates(subset=["time"], keep="last")
            df = df.sort_values("time")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_PATH, index=False)
        if existing is not None:
            added = max(0, len(df) - existing_count)
            if RESYNC_FULL_RANGE:
                print(f"Saved {len(df)} rows to {OUT_PATH} (resynced +{added})")
            else:
                print(f"Saved {len(df)} rows to {OUT_PATH} (appended +{added})")
        else:
            print(f"Saved {len(df)} rows to {OUT_PATH}")
        return True
    finally:
        mt5.shutdown()


def main() -> int:
    return 0 if download_data() else 1


if __name__ == "__main__":
    raise SystemExit(main())
