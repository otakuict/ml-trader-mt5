import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_price_data

REFRESH_FROM_MT5 = True

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "gold-data.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
LOG_DIR = ROOT_DIR / "logs"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["volatility_10"] = df["return"].rolling(10).std()
    return df


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
    df = make_features(df)
    df["target"] = df["close"].shift(-1) - df["close"]
    df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "return",
        "hl_range",
        "oc_change",
        "ma_5",
        "ma_10",
        "volatility_10",
    ]

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train[feature_cols], train["target"])

    test_score = model.score(test[feature_cols], test["target"])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"model_{timestamp}.joblib"
    model_path = MODEL_DIR / model_name
    joblib.dump(model, model_path)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "feature_cols": feature_cols,
        "rows": len(df),
        "test_score": test_score,
        "model_path": str(model_path.name),
        "trained_at": timestamp,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    logging.info("Saved model to %s", model_path)
    logging.info("Saved latest model to %s", MODEL_PATH)
    logging.info("Rows: %s", len(df))
    logging.info("Test R^2: %.4f", test_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
