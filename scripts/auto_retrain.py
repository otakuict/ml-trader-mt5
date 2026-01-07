import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_price_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold-data.csv"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

MIN_IMPROVEMENT = 0.01


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["volatility_10"] = df["return"].rolling(10).std()
    return df


def train_and_score(df: pd.DataFrame) -> tuple[Pipeline, float, list[str]]:
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
    return model, test_score, feature_cols


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Missing data file: {DATA_PATH}")
        return 1

    df = load_price_data(DATA_PATH)
    model, test_score, feature_cols = train_and_score(df)

    prev_score = None
    if META_PATH.exists():
        try:
            prev_score = float(json.loads(META_PATH.read_text()).get("test_score"))
        except (ValueError, TypeError, json.JSONDecodeError):
            prev_score = None

    print(f"New model test R^2: {test_score:.4f}")
    if prev_score is not None:
        print(f"Current model test R^2: {prev_score:.4f}")

    should_replace = prev_score is None or (test_score - prev_score) >= MIN_IMPROVEMENT
    if not should_replace:
        print("No improvement. Keeping current model.")
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_model = MODEL_DIR / f"model_{timestamp}.joblib"
    archive_meta = MODEL_DIR / f"model_{timestamp}.json"

    joblib.dump(model, MODEL_PATH)
    joblib.dump(model, archive_model)
    meta = {
        "feature_cols": feature_cols,
        "rows": int(len(df)),
        "test_score": float(test_score),
        "created_at": timestamp,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    archive_meta.write_text(json.dumps(meta, indent=2))

    print("Model updated.")
    print(f"Saved current model to {MODEL_PATH}")
    print(f"Archived model to {archive_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
