import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader import load_price_data

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gold-data.csv"
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


def main() -> int:
    if not DATA_PATH.exists():
        print(f"Missing data file: {DATA_PATH}")
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
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "feature_cols": feature_cols,
        "rows": len(df),
        "test_score": test_score,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"Saved model to {MODEL_PATH}")
    print(f"Test R^2: {test_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
