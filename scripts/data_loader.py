from pathlib import Path

import pandas as pd


def _detect_sep(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def _normalize(name: str) -> str:
    return name.strip().lower().strip("<>")


def load_price_data(path: Path) -> pd.DataFrame:
    """Load MT5-style CSVs from either MT5 export or the download script."""
    sep = _detect_sep(path)
    df = pd.read_csv(path, sep=sep)

    rename = {}
    for col in df.columns:
        norm = _normalize(col)
        if norm in ("date", "time", "datetime"):
            rename[col] = "time"
        elif norm in ("open", "high", "low", "close"):
            rename[col] = norm
        elif norm in ("tickvol", "tick_volume", "volume", "vol"):
            rename[col] = "volume"

    if rename:
        df = df.rename(columns=rename)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    return df


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        raise ValueError("Missing time column for daily resample.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df.sort_values("time").set_index("time")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg["volume"] = "sum"

    daily = df.resample("1D").agg(agg)
    daily = daily.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return daily
