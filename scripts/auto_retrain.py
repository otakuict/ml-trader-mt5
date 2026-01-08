import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from data_loader import load_price_data

REFRESH_FROM_MT5 = True
HORIZON_BARS = 6

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "gold-data-h1.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
LOG_DIR = ROOT_DIR / "logs"

MIN_IMPROVEMENT = 0.01


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        dt = pd.to_datetime(df["time"], errors="coerce")
        hour = dt.dt.hour
        dow = dt.dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
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


def train_and_score(df: pd.DataFrame) -> tuple[dict, float, float, list[str]]:
    df = make_features(df)
    df["target"] = (df["close"].shift(-HORIZON_BARS) - df["close"]) > 0
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "return_1",
        "return_3",
        "return_6",
        "hl_range",
        "oc_change",
        "ma_5",
        "ma_10",
        "ma_20",
        "ema_5",
        "ema_10",
        "volatility_10",
        "volatility_20",
        "rsi_14",
    ]
    if "time" in df.columns:
        feature_cols.extend(["hour_sin", "hour_cos", "dow_sin", "dow_cos"])
    if "volume" in df.columns:
        feature_cols.extend(["vol_ma_20", "vol_change"])
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feature_cols].to_numpy())
    model = SGDClassifier(
        loss="log_loss",
        alpha=0.0005,
        max_iter=1000,
        tol=1e-3,
        class_weight=None,
        random_state=42,
    )
    y_train = train["target"].astype(int).to_numpy()
    classes = np.array([0, 1])
    unique_classes = np.unique(y_train)
    if unique_classes.size < 2:
        sample_weight = np.ones_like(y_train, dtype=float)
    else:
        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        class_weight_map = {cls: wt for cls, wt in zip(classes, class_weights)}
        sample_weight = np.array([class_weight_map[y] for y in y_train], dtype=float)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    X_test = scaler.transform(test[feature_cols].to_numpy())
    preds = model.predict(X_test)
    test_acc = accuracy_score(test["target"], preds)
    test_prec = precision_score(test["target"], preds, zero_division=0)
    return {"scaler": scaler, "model": model}, test_acc, test_prec, feature_cols


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"auto_retrain_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Auto-retrain start")

    if REFRESH_FROM_MT5:
        try:
            from download_mt5_data import download_data
        except Exception as exc:
            logging.warning("Failed to load MT5 downloader: %s", exc)
        else:
            if not download_data():
                logging.warning("MT5 download failed; using existing data file.")

    if not DATA_PATH.exists():
        logging.error("Missing data file: %s", DATA_PATH)
        return 1

    df = load_price_data(DATA_PATH)
    if "time" not in df.columns:
        logging.error("Missing time/date column in CSV.")
        return 1
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time")
    model_bundle, test_acc, test_prec, feature_cols = train_and_score(df)

    prev_score = None
    if META_PATH.exists():
        try:
            prev_score = float(json.loads(META_PATH.read_text()).get("test_accuracy"))
        except (ValueError, TypeError, json.JSONDecodeError):
            prev_score = None

    logging.info("New model test accuracy: %.4f", test_acc)
    logging.info("New model test precision: %.4f", test_prec)
    if prev_score is not None:
        logging.info("Current model test accuracy: %.4f", prev_score)

    should_replace = prev_score is None or (test_acc - prev_score) >= MIN_IMPROVEMENT
    if not should_replace:
        logging.info("No improvement. Keeping current model.")
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    archive_model = MODEL_DIR / f"model_{timestamp}.joblib"
    archive_meta = MODEL_DIR / f"model_{timestamp}.json"

    joblib.dump(model_bundle, MODEL_PATH)
    joblib.dump(model_bundle, archive_model)
    latest_time = None
    if "time" in df.columns and not df["time"].isna().all():
        latest_time = pd.to_datetime(df["time"].iloc[-1], errors="coerce")
    meta = {
        "feature_cols": feature_cols,
        "rows": int(len(df)),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_prec),
        "created_at": timestamp,
        "model_path": str(archive_model.name),
        "model_algo": "SGDClassifier",
        "model_type": "classification",
        "target": f"next_{HORIZON_BARS}_close_up",
        "train_mode": "full",
        "last_train_time": latest_time.isoformat() if latest_time is not None else None,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    archive_meta.write_text(json.dumps(meta, indent=2))

    logging.info("Model updated.")
    logging.info("Saved current model to %s", MODEL_PATH)
    logging.info("Archived model to %s", archive_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
