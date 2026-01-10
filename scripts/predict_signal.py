import json

import joblib
import numpy as np
import pandas as pd

from data_loader import load_price_data, to_daily
from settings import DATA_PATH, META_PATH, MODEL_PATH


DAILY_TRADE_ONLY = True
DAILY_TRADE_DIR_THRESHOLD = 0.50
DAILY_TRADE_IGNORE_TREND = True
DAILY_USE_MODEL = False
DAILY_CONF_BUY = 0.65
DAILY_CONF_SELL = 0.35
TREND_MA_PERIOD = 50
TREND_MA_COL = f"ma_{TREND_MA_PERIOD}"


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


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Missing data file: {DATA_PATH}")
        return 1
    if not MODEL_PATH.exists() or not META_PATH.exists():
        print("Model not found. Run train_model.py first.")
        return 1

    df = load_price_data(DATA_PATH)
    df = to_daily(df)
    df = add_atr(make_features(df))
    df = df.dropna().reset_index(drop=True)

    meta = json.loads(META_PATH.read_text())
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
        proba = float(model.predict_proba(X)[0][1])
        pred_delta = proba
    else:
        pred_delta = float(model.predict(X)[0])

    atr = float(last.get("atr", 0.0))
    if model_type == "classification":
        ma_trend = float(last.get(TREND_MA_COL, float("nan")))
        close = float(last.get("close", float("nan")))
        trend_blocked = False
        if DAILY_TRADE_ONLY:
            if not DAILY_USE_MODEL:
                if not np.isfinite(ma_trend) or not np.isfinite(close):
                    signal = "HOLD"
                    trend_blocked = True
                else:
                    signal = "BUY" if close >= ma_trend else "SELL"
                print(f"Daily trade mode: trend (close vs MA{TREND_MA_PERIOD})")
            else:
                if pred_delta >= DAILY_CONF_BUY:
                    signal = "BUY"
                elif pred_delta <= DAILY_CONF_SELL:
                    signal = "SELL"
                else:
                    if DAILY_TRADE_IGNORE_TREND:
                        signal = "BUY" if pred_delta >= DAILY_TRADE_DIR_THRESHOLD else "SELL"
                    else:
                        if not np.isfinite(ma_trend) or not np.isfinite(close):
                            signal = "HOLD"
                            trend_blocked = True
                        else:
                            signal = "BUY" if close >= ma_trend else "SELL"
                print(f"Predicted up probability: {pred_delta:.4f}")
                print(
                    f"Daily conf: BUY>= {DAILY_CONF_BUY:.2f} SELL<= {DAILY_CONF_SELL:.2f}"
                )
        else:
            if pred_delta >= 0.85:
                signal = "BUY"
            elif pred_delta <= 0.15:
                signal = "SELL"
            else:
                signal = "HOLD"
            if np.isfinite(ma_trend) and np.isfinite(close):
                if signal == "BUY" and close <= ma_trend:
                    signal = "HOLD"
                    trend_blocked = True
                elif signal == "SELL" and close >= ma_trend:
                    signal = "HOLD"
                    trend_blocked = True
            else:
                if signal in {"BUY", "SELL"}:
                    signal = "HOLD"
                    trend_blocked = True
            print(f"Predicted up probability: {pred_delta:.4f}")
            print("Thresholds: BUY>=0.85 SELL<=0.15")
        if trend_blocked:
            print(f"Trend filter blocked trade (close vs MA{TREND_MA_PERIOD}).")
    else:
        close = float(last["close"])
        threshold = max(atr * 0.25, close * 0.002)
        if pred_delta > threshold:
            signal = "BUY"
        elif pred_delta < -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        print(f"Predicted next-bar delta: {pred_delta:.4f}")
        print(f"ATR: {atr:.4f} | Threshold: {threshold:.4f}")
    print(f"Signal: {signal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
