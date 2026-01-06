import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path

SYMBOL = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_D1
BARS = 2000
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "gold_d1.csv"


def main() -> int:
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return 1
    try:
        rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
        if rates is None or len(rates) == 0:
            print("No data returned. Check symbol name and history in MT5.")
            return 1
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_PATH, index=False)
        print(f"Saved {len(df)} rows to {OUT_PATH}")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
