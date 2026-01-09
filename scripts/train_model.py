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

from data_loader import load_price_data, to_daily

REFRESH_FROM_MT5 = True
INCREMENTAL_TRAIN = True
HORIZON_DAYS = 1

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "gold-data-h4.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
LOG_DIR = ROOT_DIR / "logs"


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


def load_model_bundle():
    if not MODEL_PATH.exists():
        return None, None
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle.get("scaler"), bundle.get("model")
    return None, None


def build_model() -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=0.0005,
        max_iter=1000,
        tol=1e-3,
        class_weight=None,
        random_state=42,
    )


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("Training start")

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
    df = to_daily(df)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time")
    df = make_features(df)
    df["target"] = (df["close"].shift(-HORIZON_DAYS) - df["close"]) > 0
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
        feature_cols.extend(["dow_sin", "dow_cos"])
    if "volume" in df.columns:
        feature_cols.extend(["vol_ma_20", "vol_change"])

    scaler, model = load_model_bundle()
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
        except (ValueError, TypeError, json.JSONDecodeError):
            meta = {}

    last_train_time = None
    if meta.get("last_train_time"):
        last_train_time = pd.to_datetime(meta["last_train_time"], errors="coerce")

    train_df = df
    train_mode = "full"
    if INCREMENTAL_TRAIN and last_train_time is not None and scaler and model:
        train_df = df[df["time"] > last_train_time]
        train_mode = "incremental"

    if train_df.empty:
        logging.info("No new data since last training. Skipping.")
        return 0

    if not scaler or not model or not isinstance(model, SGDClassifier):
        scaler = StandardScaler()
        model = build_model()
        train_df = df
        train_mode = "full"

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["target"].astype(int).to_numpy()
    scaler.partial_fit(X_train)
    X_train = scaler.transform(X_train)

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

    if not hasattr(model, "classes_"):
        model.partial_fit(X_train, y_train, classes=classes, sample_weight=sample_weight)
    else:
        model.partial_fit(X_train, y_train, sample_weight=sample_weight)

    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:]
    if test.empty:
        logging.warning("No test split available for metrics.")
        test_acc = 0.0
        test_prec = 0.0
    else:
        X_test = scaler.transform(test[feature_cols].to_numpy())
        preds = model.predict(X_test)
        test_acc = accuracy_score(test["target"], preds)
        test_prec = precision_score(test["target"], preds, zero_division=0)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"model_{timestamp}.joblib"
    model_path = MODEL_DIR / model_name
    bundle = {"scaler": scaler, "model": model}
    joblib.dump(bundle, model_path)
    joblib.dump(bundle, MODEL_PATH)

    latest_time = df["time"].iloc[-1]
    meta = {
        "feature_cols": feature_cols,
        "rows": len(df),
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "model_path": str(model_path.name),
        "trained_at": timestamp,
        "model_algo": "SGDClassifier",
        "model_type": "classification",
        "target": f"next_{HORIZON_DAYS}_close_up",
        "train_mode": train_mode,
        "last_train_time": latest_time.isoformat(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    logging.info("Saved model to %s", model_path)
    logging.info("Saved latest model to %s", MODEL_PATH)
    logging.info("Rows: %s", len(df))
    logging.info("Train mode: %s", train_mode)
    logging.info("Last train time: %s", latest_time.isoformat())
    logging.info("Test accuracy: %.4f", test_acc)
    logging.info("Test precision: %.4f", test_prec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
