import logging
from datetime import datetime

import backtrader as bt

from data_loader import load_price_data
from settings import DATA_PATH, LOG_DIR



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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"backtest_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Backtest start")

    if not DATA_PATH.exists():
        logging.error("Missing data file: %s", DATA_PATH)
        return 1

    df = load_price_data(DATA_PATH)
    if "time" not in df.columns:
        logging.error("Missing time/date column in CSV.")
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

    pnl = end_value - start_value
    logging.info("Start value: %.2f", start_value)
    logging.info("End value:   %.2f", end_value)
    logging.info("PnL:         %.2f", pnl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
