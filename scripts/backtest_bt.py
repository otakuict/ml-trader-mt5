import backtrader as bt
from pathlib import Path

from data_loader import load_price_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold_d1.csv"


class SmaCross(bt.Strategy):
    params = dict(fast=10, slow=30)

    def __init__(self):
        fast = bt.ind.SMA(period=self.p.fast)
        slow = bt.ind.SMA(period=self.p.slow)
        self.cross = bt.ind.CrossOver(fast, slow)

    def next(self):
        if not self.position and self.cross > 0:
            self.buy()
        elif self.position and self.cross < 0:
            self.sell()


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Missing data file: {DATA_PATH}")
        return 1

    df = load_price_data(DATA_PATH)
    if "time" not in df.columns:
        print("Missing time/date column in CSV.")
        return 1

    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    df = df.sort_values("time")
    df = df.set_index("time")

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0005)

    start_value = cerebro.broker.getvalue()
    cerebro.run()
    end_value = cerebro.broker.getvalue()

    print(f"Start value: {start_value:.2f}")
    print(f"End value:   {end_value:.2f}")
    print(f"PnL:         {end_value - start_value:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
