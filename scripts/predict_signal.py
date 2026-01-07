import json
from pathlib import Path

import joblib
import pandas as pd

from data_loader import load_price_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold_d1.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
META_PATH = Path(__file__).resolve().parents[1] / "models" / "model_meta.json"


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


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Missing data file: {DATA_PATH}")
        return 1
    if not MODEL_PATH.exists() or not META_PATH.exists():
        print("Model not found. Run train_model.py first.")
        return 1

    df = load_price_data(DATA_PATH)
    df = add_atr(make_features(df))
    df = df.dropna().reset_index(drop=True)

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

    print(f"Predicted next-day delta: {pred_delta:.4f}")
    print(f"ATR: {atr:.4f} | Threshold: {threshold:.4f}")
    print(f"Signal: {signal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
