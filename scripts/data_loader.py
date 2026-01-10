from pathlib import Path

import pandas as pd


def _detect_sep(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first_line = handle.readline()
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    return ","


def _normalize(name: str) -> str:
    return name.strip().lower().strip("<>")


def load_price_data(path: Path) -> pd.DataFrame:
    """Load MT5-style CSVs from either MT5 export or the download script."""
    sep = _detect_sep(path)
    df = pd.read_csv(path, sep=sep)
    if len(df.columns) == 1 and sep == "," and "\t" in str(df.columns[0]):
        df = pd.read_csv(path, sep="\t")

    rename = {}
    date_col = None
    time_col = None
    datetime_col = None
    for col in df.columns:
        norm = _normalize(col)
        if norm == "date":
            date_col = col
        elif norm == "time":
            time_col = col
        elif norm == "datetime":
            datetime_col = col
        elif norm in ("open", "high", "low", "close"):
            rename[col] = norm
        elif norm in ("tickvol", "tick_volume", "volume", "vol"):
            rename[col] = "volume"

    if datetime_col:
        rename[datetime_col] = "time"
    elif date_col and time_col:
        df["time"] = (
            df[date_col].astype(str).str.strip()
            + " "
            + df[time_col].astype(str).str.strip()
        )
        rename.pop(date_col, None)
        rename.pop(time_col, None)
    elif date_col:
        rename[date_col] = "time"

    if rename:
        df = df.rename(columns=rename)

    if date_col and time_col:
        cols_to_drop = [col for col in (date_col, time_col) if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    if (df.columns == "time").sum() > 1:
        time_df = df.loc[:, df.columns == "time"]
        df = df.drop(columns=time_df.columns)
        df["time"] = time_df.bfill(axis=1).iloc[:, 0]

    if (df.columns == "volume").sum() > 1:
        vol_df = df.loc[:, df.columns == "volume"]
        df = df.drop(columns=vol_df.columns)
        df["volume"] = vol_df.bfill(axis=1).iloc[:, 0]

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
