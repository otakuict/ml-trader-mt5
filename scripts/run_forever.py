import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from data_loader import load_price_data, to_daily

SYMBOL = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_H4
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
ML_TRADING_ENABLED = True
USE_ATR_SLTP = True
SL_ATR_MULT = 1.2
TP_ATR_MULT = 4.0
CLASS_BUY_THRESHOLD = 0.85
CLASS_SELL_THRESHOLD = 0.15
TREND_FILTER = True
FORCE_DAILY_TRADE = True
DAILY_TRADE_HOUR = 23
DAILY_TRADE_MINUTE = 55
DAILY_TRADE_DIR_THRESHOLD = 0.50
DAILY_TRADE_IGNORE_TREND = True
DAILY_TRADE_ONLY = True
DAILY_CLOSE_EXISTING = True
DAILY_FALLBACK_TREND = True
DAILY_CONF_BUY = 0.65
DAILY_CONF_SELL = 0.35
DAILY_USE_MODEL = False
TREND_MA_PERIOD = 50
TREND_MA_COL = f"ma_{TREND_MA_PERIOD}"

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold-data-h4.csv"
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
    if "time" in df.columns:
        dt = pd.to_datetime(df["time"], errors="coerce")
        dow = dt.dt.dayofweek
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["return_1"] = df["close"].pct_change()
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df[TREND_MA_COL] = df["close"].rolling(TREND_MA_PERIOD).mean()
    df["ema_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["ema_10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    if "volume" in df.columns:
        df["vol_ma_20"] = df["volume"].rolling(20).mean()
        df["vol_change"] = df["volume"].pct_change().replace([pd.NA, pd.NaT], 0.0)
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


def compute_signal(apply_trend_filter: bool = True) -> Optional[dict]:
    if not DATA_PATH.exists():
        print(f"Missing data file: {DATA_PATH}")
        return None
    if not MODEL_PATH.exists() or not META_PATH.exists():
        print("Model not found. Run train_model.py first.")
        return None

    df = load_price_data(DATA_PATH)
    df = to_daily(df)
    df = add_atr(make_features(df))
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        print("Not enough data to compute signal.")
        return None

    meta = json.loads(META_PATH.read_text())
    model_name = meta.get("model_path", MODEL_PATH.name)
    print(f"Using model: {model_name}")
    model_type = meta.get("model_type", "regression")
    feature_cols = meta["feature_cols"]
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        scaler = bundle.get("scaler")
    else:
        model = bundle
        scaler = None

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    last = df.iloc[-1]
    X = last[feature_cols].to_frame().T
    if scaler is not None:
        X = scaler.transform(X.to_numpy())
    if model_type == "classification":
        pred_proba = float(model.predict_proba(X)[0][1])
        if pred_proba >= CLASS_BUY_THRESHOLD:
            signal = "BUY"
        elif pred_proba <= CLASS_SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"
        pred_delta = pred_proba
        threshold = None
        atr = float(last.get("atr", 0.0))
    else:
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

    trend_blocked = False
    close = float(last.get("close", float("nan")))
    ma_trend = float(last.get(TREND_MA_COL, float("nan")))
    if apply_trend_filter and TREND_FILTER and signal in {"BUY", "SELL"}:
        if not np.isfinite(ma_trend) or not np.isfinite(close):
            signal = "HOLD"
            trend_blocked = True
        elif signal == "BUY" and close <= ma_trend:
            signal = "HOLD"
            trend_blocked = True
        elif signal == "SELL" and close >= ma_trend:
            signal = "HOLD"
            trend_blocked = True

    return {
        "signal": signal,
        "pred_delta": pred_delta,
        "atr": atr,
        "threshold": threshold,
        "model_type": model_type,
        "trend_blocked": trend_blocked,
        "close": close,
        "ma_trend": ma_trend,
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


def compute_sl_tp(order_type: int, price: float, atr: float, symbol_info) -> tuple:
    if not USE_ATR_SLTP or not np.isfinite(atr) or atr <= 0:
        return None, None
    sl_dist = atr * SL_ATR_MULT
    tp_dist = atr * TP_ATR_MULT
    if order_type == mt5.ORDER_TYPE_BUY:
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist
    digits = getattr(symbol_info, "digits", 5)
    sl = round(sl, digits)
    tp = round(tp, digits)
    return sl, tp


def place_order(signal: str, atr: float | None = None) -> bool:
    if signal not in {"BUY", "SELL", "HOLD"}:
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
        sl, tp = compute_sl_tp(order_type, price, float(atr or 0.0), symbol_info)

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
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        result = mt5.order_send(request)
        if result is None:
            print("Order send failed.")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: retcode={result.retcode} comment={result.comment}")
            return False

        if sl is not None and tp is not None:
            print(f"Order placed: {signal} {volume} {SYMBOL} at {price} sl={sl} tp={tp}")
        else:
            print(f"Order placed: {signal} {volume} {SYMBOL} at {price}")
        return True
    finally:
        mt5.shutdown()


def main() -> int:
    last_seen = None
    last_daily_trade_date = None
    print("Running forever. Press Ctrl+C to stop.")
    while True:
        if download_data():
            current = latest_bar_time()
            if not current:
                time.sleep(SLEEP_SECONDS)
                continue

            now = datetime.now()
            if not DAILY_TRADE_ONLY and current == last_seen:
                time.sleep(SLEEP_SECONDS)
                continue

            last_seen = current
            print(f"New H4 candle detected at {current} | Local time: {now}")
            if DRY_RUN:
                print("DRY_RUN is on. No orders will be sent.")
            else:
                if not ML_TRADING_ENABLED:
                    print("ML trading disabled. Set ML_TRADING_ENABLED = True to trade.")
                    time.sleep(SLEEP_SECONDS)
                    continue
                signal_data = compute_signal(apply_trend_filter=not DAILY_TRADE_ONLY)
                if not signal_data:
                    time.sleep(SLEEP_SECONDS)
                    continue
                if signal_data["model_type"] == "classification":
                    if DAILY_TRADE_ONLY:
                        if DAILY_USE_MODEL:
                            print(
                                "Signal:",
                                f"{signal_data['signal']}",
                                f"prob={signal_data['pred_delta']:.4f}",
                                f"atr={signal_data['atr']:.4f}",
                                f"daily_conf={DAILY_CONF_BUY:.2f}/{DAILY_CONF_SELL:.2f}",
                            )
                        else:
                            print(
                                "Signal:",
                                f"{signal_data['signal']}",
                                f"atr={signal_data['atr']:.4f}",
                                "daily_mode=trend",
                            )
                    else:
                        print(
                            "Signal:",
                            f"{signal_data['signal']}",
                            f"prob={signal_data['pred_delta']:.4f}",
                            f"atr={signal_data['atr']:.4f}",
                            f"thresholds={CLASS_BUY_THRESHOLD:.2f}/{CLASS_SELL_THRESHOLD:.2f}",
                        )
                else:
                    print(
                        "Signal:",
                        f"{signal_data['signal']}",
                        f"pred={signal_data['pred_delta']:.4f}",
                        f"atr={signal_data['atr']:.4f}",
                        f"threshold={signal_data['threshold']:.4f}",
                    )
                if DAILY_TRADE_ONLY:
                    if (
                        last_daily_trade_date != now.date()
                        and (
                            now.hour > DAILY_TRADE_HOUR
                            or (
                                now.hour == DAILY_TRADE_HOUR
                                and now.minute >= DAILY_TRADE_MINUTE
                            )
                        )
                    ):
                        if signal_data["model_type"] == "classification":
                            if not DAILY_USE_MODEL:
                                if not np.isfinite(signal_data["ma_trend"]):
                                    print(
                                        f"Daily trade skipped: MA{TREND_MA_PERIOD} not available."
                                    )
                                    time.sleep(SLEEP_SECONDS)
                                    continue
                                forced_signal = (
                                    "BUY"
                                    if signal_data["close"] >= signal_data["ma_trend"]
                                    else "SELL"
                                )
                            elif signal_data["pred_delta"] >= DAILY_CONF_BUY:
                                forced_signal = "BUY"
                            elif signal_data["pred_delta"] <= DAILY_CONF_SELL:
                                forced_signal = "SELL"
                            elif DAILY_FALLBACK_TREND:
                                forced_signal = (
                                    "BUY"
                                    if signal_data["close"] >= signal_data["ma_trend"]
                                    else "SELL"
                                )
                            else:
                                forced_signal = (
                                    "BUY"
                                    if signal_data["pred_delta"] >= DAILY_TRADE_DIR_THRESHOLD
                                    else "SELL"
                                )
                        else:
                            forced_signal = (
                                "BUY" if signal_data["pred_delta"] >= 0 else "SELL"
                            )

                        if not DAILY_TRADE_IGNORE_TREND:
                            close = signal_data.get("close", float("nan"))
                            ma_trend = signal_data.get("ma_trend", float("nan"))
                            if (
                                not np.isfinite(close)
                                or not np.isfinite(ma_trend)
                                or (forced_signal == "BUY" and close <= ma_trend)
                                or (forced_signal == "SELL" and close >= ma_trend)
                            ):
                                print("Daily trade blocked by trend filter.")
                                time.sleep(SLEEP_SECONDS)
                                continue

                        if DAILY_CLOSE_EXISTING:
                            place_order("HOLD", signal_data["atr"])

                        print(f"Daily trade at {now} -> {forced_signal}")
                        trade_made = place_order(forced_signal, signal_data["atr"])
                        if trade_made:
                            last_daily_trade_date = now.date()
                else:
                    trade_made = False
                    if signal_data["signal"] == "HOLD":
                        if signal_data.get("trend_blocked"):
                            print("No trade: blocked by trend filter.")
                        else:
                            print("No trade: signal is HOLD.")
                        if CLOSE_ON_HOLD:
                            place_order("HOLD", signal_data["atr"])
                    else:
                        trade_made = place_order(
                            signal_data["signal"], signal_data["atr"]
                        )

                    if trade_made:
                        last_daily_trade_date = now.date()
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
