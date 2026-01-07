import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_D1
BARS = 2000
SLEEP_SECONDS = 300
DRY_RUN = False
LOT_SIZE = 0.01
MAX_POSITIONS = 1
MAGIC = 51001
DEVIATION = 20
ALLOW_REAL = False
CLOSE_ON_OPPOSITE = True
CLOSE_ON_HOLD = True

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold_d1.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
META_PATH = Path(__file__).resolve().parents[1] / "models" / "model_meta.json"


def download_data() -> bool:
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return False
    try:
        version = mt5.version()
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        print(f"MT5 connected | version={version}")
        if terminal:
            terminal_server = getattr(terminal, "server", None)
            print(
                "Terminal:",
                f"path={terminal.path}",
                f"server={terminal_server}",
                f"connected={terminal.connected}",
            )
        if account:
            print(
                "Account:",
                f"login={account.login}",
                f"server={account.server}",
                f"balance={account.balance}",
            )
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


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["volatility_10"] = df["return"].rolling(10).std()
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df


def compute_signal() -> Optional[dict]:
    if not DATA_PATH.exists():
        print(f"Missing data file: {DATA_PATH}")
        return None
    if not MODEL_PATH.exists() or not META_PATH.exists():
        print("Model not found. Run train_model.py first.")
        return None

    df = pd.read_csv(DATA_PATH)
    df = add_atr(make_features(df))
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        print("Not enough data to compute signal.")
        return None

    meta = json.loads(META_PATH.read_text())
    feature_cols = meta["feature_cols"]
    model = joblib.load(MODEL_PATH)

    last = df.iloc[-1]
    X = last[feature_cols].to_frame().T
    pred_delta = float(model.predict(X)[0])

    atr = float(last.get("atr", 0.0))
    close = float(last["close"])
    threshold = max(atr * 0.25, close * 0.002)

    if pred_delta > threshold:
        signal = "BUY"
    elif pred_delta < -threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "pred_delta": pred_delta,
        "atr": atr,
        "threshold": threshold,
    }


def normalize_volume(volume: float, symbol_info) -> float:
    min_vol = getattr(symbol_info, "volume_min", volume)
    max_vol = getattr(symbol_info, "volume_max", volume)
    step = getattr(symbol_info, "volume_step", 0.01)
    vol = max(volume, min_vol)
    vol = min(vol, max_vol)
    if step:
        vol = round(vol / step) * step
        if vol < min_vol:
            vol = min_vol
    return float(vol)


def resolve_filling(symbol_info) -> int:
    filling = mt5.ORDER_FILLING_IOC
    if hasattr(symbol_info, "filling_mode"):
        filling_mode = symbol_info.filling_mode
        if hasattr(mt5, "SYMBOL_FILLING_FOK") and filling_mode == mt5.SYMBOL_FILLING_FOK:
            filling = mt5.ORDER_FILLING_FOK
        elif hasattr(mt5, "SYMBOL_FILLING_IOC") and filling_mode == mt5.SYMBOL_FILLING_IOC:
            filling = mt5.ORDER_FILLING_IOC
        elif (
            hasattr(mt5, "SYMBOL_FILLING_RETURN")
            and filling_mode == mt5.SYMBOL_FILLING_RETURN
        ):
            filling = mt5.ORDER_FILLING_RETURN
    return filling


def close_position(symbol_info, position) -> bool:
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("No tick data; cannot close position.")
        return False
    close_type = (
        mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    )
    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
    filling = resolve_filling(symbol_info)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": close_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "ml-trader-close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling,
        "position": position.ticket,
    }
    result = mt5.order_send(request)
    if result is None:
        print("Close failed: order_send returned None.")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close failed: retcode={result.retcode} comment={result.comment}")
        return False
    print(f"Closed position {position.ticket} for {SYMBOL}")
    return True


def place_order(signal: str) -> bool:
    if signal not in {"BUY", "SELL"}:
        return False
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return False
    try:
        account = mt5.account_info()
        if not account:
            print("Account info unavailable.")
            return False
        if (
            getattr(account, "trade_mode", None) != mt5.ACCOUNT_TRADE_MODE_DEMO
            and not ALLOW_REAL
        ):
            print("Non-demo account detected. Set ALLOW_REAL = True to trade.")
            return False

        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"Symbol not found: {SYMBOL}")
            return False
        if not symbol_info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                print(f"Failed to select symbol: {SYMBOL}")
                return False

        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            positions = []

        if signal == "HOLD" and CLOSE_ON_HOLD:
            for pos in positions:
                close_position(symbol_info, pos)
            return True

        if CLOSE_ON_OPPOSITE and positions:
            for pos in positions:
                if signal == "BUY" and pos.type == mt5.ORDER_TYPE_SELL:
                    close_position(symbol_info, pos)
                elif signal == "SELL" and pos.type == mt5.ORDER_TYPE_BUY:
                    close_position(symbol_info, pos)

        positions = mt5.positions_get(symbol=SYMBOL) or []
        same_dir = [
            pos
            for pos in positions
            if (signal == "BUY" and pos.type == mt5.ORDER_TYPE_BUY)
            or (signal == "SELL" and pos.type == mt5.ORDER_TYPE_SELL)
        ]
        if len(same_dir) >= MAX_POSITIONS:
            print(f"Open position exists for {SYMBOL}; skipping new order.")
            return False

        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            print("No tick data; cannot place order.")
            return False

        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if signal == "BUY" else tick.bid
        volume = normalize_volume(LOT_SIZE, symbol_info)

        filling = resolve_filling(symbol_info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": DEVIATION,
            "magic": MAGIC,
            "comment": "ml-trader",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result is None:
            print("Order send failed.")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: retcode={result.retcode} comment={result.comment}")
            return False

        print(f"Order placed: {signal} {volume} {SYMBOL} at {price}")
        return True
    finally:
        mt5.shutdown()


def main() -> int:
    last_seen = None
    print("Running forever. Press Ctrl+C to stop.")
    while True:
        if download_data():
            current = latest_bar_time()
            if current and current != last_seen:
                last_seen = current
                now = datetime.now()
                print(f"New D1 candle detected at {current} | Local time: {now}")
                if DRY_RUN:
                    print("DRY_RUN is on. No orders will be sent.")
                else:
                    signal_data = compute_signal()
                    if not signal_data:
                        time.sleep(SLEEP_SECONDS)
                        continue
                    print(
                        "Signal:",
                        f"{signal_data['signal']}",
                        f"pred={signal_data['pred_delta']:.4f}",
                        f"atr={signal_data['atr']:.4f}",
                        f"threshold={signal_data['threshold']:.4f}",
                    )
                    if signal_data["signal"] == "HOLD":
                        print("No trade: signal is HOLD.")
                        if CLOSE_ON_HOLD:
                            place_order("HOLD")
                    else:
                        place_order(signal_data["signal"])
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
