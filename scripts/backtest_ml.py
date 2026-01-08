import json
import logging
from datetime import datetime
from pathlib import Path

import backtrader as bt
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from data_loader import load_price_data

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "gold-data-h1.csv"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
META_PATH = ROOT_DIR / "models" / "model_meta.json"
LOG_DIR = ROOT_DIR / "logs"

BUY_THRESHOLD = 0.85
SELL_THRESHOLD = 0.15
USE_WALK_FORWARD = True
TRAIN_WINDOW = 5000
TEST_WINDOW = 500
USE_ATR_SLTP = True
SL_ATR_MULT = 1.5
TP_ATR_MULT = 3.0
CLOSE_ON_TREND_FILTER = True
CLOSE_ON_HOLD = True
CLOSE_ON_OPPOSITE = True
START_CASH = 10000.0
COMMISSION = 0.0005
TRADE_SIZE = 1
FORCE_DAILY_TRADE = True
DAILY_TRADE_HOUR = 9
DAILY_TRADE_MINUTE = 0
DAILY_TRADE_DIR_THRESHOLD = 0.50
DAILY_TRADE_IGNORE_TREND = True


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        dt = pd.to_datetime(df["time"], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        dt = pd.to_datetime(df.index, errors="coerce")
    else:
        dt = None
    if dt is not None:
        if isinstance(dt, pd.DatetimeIndex):
            hour = dt.hour
            dow = dt.dayofweek
        else:
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


def compute_signals(
    df: pd.DataFrame, model, scaler, feature_cols: list[str], model_type: str
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    features = make_features(df)
    if model_type != "classification":
        features = add_atr(features)

    signals = pd.Series(0, index=features.index, dtype=int)
    prob = pd.Series(np.nan, index=features.index, dtype=float)
    mask = features[feature_cols].notna().all(axis=1)
    if not mask.any():
        return signals, prob, features

    X = features.loc[mask, feature_cols]
    idx = X.index
    if scaler is not None:
        X = scaler.transform(X.to_numpy())
    if model_type == "classification":
        proba = model.predict_proba(X)[:, 1]
        prob.loc[idx] = proba
        close = features.loc[idx, "close"]
        ma_20 = features.loc[idx, "ma_20"]
        sig = pd.Series(0, index=idx, dtype=int)
        buy_mask = proba >= BUY_THRESHOLD
        sell_mask = proba <= SELL_THRESHOLD
        if CLOSE_ON_TREND_FILTER:
            buy_mask &= close > ma_20
            sell_mask &= close < ma_20
        sig[buy_mask] = 1
        sig[sell_mask] = -1
        signals.loc[sig.index] = sig
    else:
        preds = model.predict(X)
        atr = features.loc[mask, "atr"]
        close = features.loc[mask, "close"]
        threshold = np.maximum(atr * 0.25, close * 0.002)
        sig = pd.Series(0, index=idx, dtype=int)
        sig[preds > threshold] = 1
        sig[preds < -threshold] = -1
        signals.loc[sig.index] = sig

    return signals, prob, features


def parse_target_horizon(meta: dict) -> int:
    target = str(meta.get("target", ""))
    for part in target.split("_"):
        if part.isdigit():
            return int(part)
    return 6


def compute_signals_walk_forward(
    df: pd.DataFrame,
    base_model,
    feature_cols: list[str],
    model_type: str,
    horizon: int,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    features_full = make_features(df)
    if model_type != "classification":
        features_full = add_atr(features_full)
    features = features_full.copy()

    if model_type == "classification":
        features["target"] = (features["close"].shift(-horizon) - features["close"]) > 0
    else:
        features["target"] = features["close"].shift(-horizon) - features["close"]

    needed_cols = list(feature_cols) + ["target"]
    features = features.dropna(subset=needed_cols)
    signals = pd.Series(0, index=df.index, dtype=int)
    prob = pd.Series(np.nan, index=df.index, dtype=float)

    total = len(features)
    if total < TRAIN_WINDOW + 1:
        return signals, prob, features_full

    start = TRAIN_WINDOW
    while start < total:
        train = features.iloc[start - TRAIN_WINDOW : start]
        test = features.iloc[start : start + TEST_WINDOW]
        if test.empty:
            break
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[feature_cols].to_numpy())
        model = clone(base_model)
        model.fit(X_train, train["target"].astype(int))
        if model_type == "classification":
            X_test = scaler.transform(test[feature_cols].to_numpy())
            proba = model.predict_proba(X_test)[:, 1]
            close = test["close"]
            ma_20 = test["ma_20"]
            sig = pd.Series(0, index=test.index, dtype=int)
            buy_mask = proba >= BUY_THRESHOLD
            sell_mask = proba <= SELL_THRESHOLD
            if CLOSE_ON_TREND_FILTER:
                buy_mask &= close > ma_20
                sell_mask &= close < ma_20
            sig[buy_mask] = 1
            sig[sell_mask] = -1
            signals.loc[sig.index] = sig
            prob.loc[sig.index] = proba
        else:
            X_test = scaler.transform(test[feature_cols].to_numpy())
            preds = model.predict(X_test)
            atr = features.loc[test.index, "atr"]
            close = features.loc[test.index, "close"]
            threshold = np.maximum(atr * 0.25, close * 0.002)
            sig = pd.Series(0, index=test.index, dtype=int)
            sig[preds > threshold] = 1
            sig[preds < -threshold] = -1
            signals.loc[sig.index] = sig
        start += TEST_WINDOW

    return signals, prob, features_full


class SignalData(bt.feeds.PandasData):
    lines = ("signal", "atr", "prob", "ma_20")
    params = (("signal", "signal"), ("atr", "atr"), ("prob", "prob"), ("ma_20", "ma_20"))


class SignalStrategy(bt.Strategy):
    params = dict(
        size=TRADE_SIZE,
        close_on_hold=CLOSE_ON_HOLD,
        close_on_opposite=CLOSE_ON_OPPOSITE,
        use_atr_sltp=USE_ATR_SLTP,
        sl_atr_mult=SL_ATR_MULT,
        tp_atr_mult=TP_ATR_MULT,
    )

    def __init__(self):
        self.signal = self.datas[0].signal
        self.atr = self.datas[0].atr
        self.prob = self.datas[0].prob
        self.ma_20 = self.datas[0].ma_20
        self.stop_price = None
        self.target_price = None
        self.last_trade_date = None
        self.trade_count = 0
        self.forced_trade_count = 0

    def _reset_risk(self) -> None:
        self.stop_price = None
        self.target_price = None

    def _set_risk(self, price: float, is_long: bool) -> None:
        if not self.p.use_atr_sltp:
            return
        atr_val = float(self.atr[0])
        if not np.isfinite(atr_val) or atr_val <= 0:
            return
        sl_dist = atr_val * self.p.sl_atr_mult
        tp_dist = atr_val * self.p.tp_atr_mult
        if is_long:
            self.stop_price = price - sl_dist
            self.target_price = price + tp_dist
        else:
            self.stop_price = price + sl_dist
            self.target_price = price - tp_dist

    def next(self):
        if self.position and self.p.use_atr_sltp and self.stop_price is not None:
            if self.position.size > 0:
                if self.data.low[0] <= self.stop_price or self.data.high[0] >= self.target_price:
                    self.close()
                    self._reset_risk()
                    return
            else:
                if self.data.high[0] >= self.stop_price or self.data.low[0] <= self.target_price:
                    self.close()
                    self._reset_risk()
                    return

        sig = int(self.signal[0])
        if sig == 0:
            if self.p.close_on_hold and self.position:
                self.close()
                self._reset_risk()
            self._maybe_force_daily_trade()
            return

        if self.position:
            if (
                sig == 1
                and self.position.size < 0
                and self.p.close_on_opposite
            ):
                self.close()
                self._reset_risk()
            elif (
                sig == -1
                and self.position.size > 0
                and self.p.close_on_opposite
            ):
                self.close()
                self._reset_risk()

        if sig == 1 and self.position.size <= 0:
            self.buy(size=self.p.size)
            self._set_risk(float(self.data.close[0]), True)
            self.last_trade_date = self.data.datetime.date(0)
            self.trade_count += 1
        elif sig == -1 and self.position.size >= 0:
            self.sell(size=self.p.size)
            self._set_risk(float(self.data.close[0]), False)
            self.last_trade_date = self.data.datetime.date(0)
            self.trade_count += 1
        self._maybe_force_daily_trade()

    def _maybe_force_daily_trade(self) -> None:
        if not FORCE_DAILY_TRADE:
            return
        current_date = self.data.datetime.date(0)
        current_time = self.data.datetime.time(0)
        if self.last_trade_date == current_date:
            return
        if current_time.hour < DAILY_TRADE_HOUR or (
            current_time.hour == DAILY_TRADE_HOUR
            and current_time.minute < DAILY_TRADE_MINUTE
        ):
            return

        if self.position:
            return

        prob = float(self.prob[0])
        if not np.isfinite(prob):
            return
        forced_signal = "BUY" if prob >= DAILY_TRADE_DIR_THRESHOLD else "SELL"
        if not DAILY_TRADE_IGNORE_TREND:
            close = float(self.data.close[0])
            ma_20 = float(self.ma_20[0])
            if not np.isfinite(ma_20):
                return
            if (forced_signal == "BUY" and close <= ma_20) or (
                forced_signal == "SELL" and close >= ma_20
            ):
                return

        if forced_signal == "BUY":
            self.buy(size=self.p.size)
            self._set_risk(float(self.data.close[0]), True)
        else:
            self.sell(size=self.p.size)
            self._set_risk(float(self.data.close[0]), False)
        self.last_trade_date = current_date
        self.trade_count += 1
        self.forced_trade_count += 1

    def stop(self) -> None:
        logging.info(
            "Trades: %s | Forced trades: %s", self.trade_count, self.forced_trade_count
        )


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"backtest_ml_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.info("ML backtest start")

    if not DATA_PATH.exists():
        logging.error("Missing data file: %s", DATA_PATH)
        return 1
    if not MODEL_PATH.exists() or not META_PATH.exists():
        logging.error("Model not found. Run train_model.py first.")
        return 1

    df = load_price_data(DATA_PATH)
    if "time" not in df.columns:
        logging.error("Missing time/date column in CSV.")
        return 1

    meta = json.loads(META_PATH.read_text())
    feature_cols = meta["feature_cols"]
    model_type = meta.get("model_type", "classification")
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        scaler = bundle.get("scaler")
    else:
        model = bundle
        scaler = None

    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    df = df.sort_values("time").set_index("time")
    df = add_atr(df)

    horizon = parse_target_horizon(meta)
    if USE_WALK_FORWARD and model_type == "classification":
        signals, prob, features = compute_signals_walk_forward(
            df, model, feature_cols, model_type, horizon
        )
    else:
        signals, prob, features = compute_signals(df, model, scaler, feature_cols, model_type)
    df["signal"] = signals
    df["prob"] = prob
    if "ma_20" in features.columns:
        df["ma_20"] = features["ma_20"]
    else:
        df["ma_20"] = np.nan

    signal_counts = df["signal"].value_counts()
    buy_signals = int(signal_counts.get(1, 0))
    sell_signals = int(signal_counts.get(-1, 0))
    total_signals = buy_signals + sell_signals

    logging.info("Using model: %s", meta.get("model_path", MODEL_PATH.name))
    logging.info("Model type: %s", model_type)
    logging.info("Walk-forward: %s", USE_WALK_FORWARD)
    if model_type == "classification":
        logging.info("Thresholds: BUY>=%.2f SELL<=%.2f", BUY_THRESHOLD, SELL_THRESHOLD)
    logging.info("Trend filter: %s", CLOSE_ON_TREND_FILTER)
    logging.info("ATR SL/TP: %s | SLx=%.2f TPx=%.2f", USE_ATR_SLTP, SL_ATR_MULT, TP_ATR_MULT)

    cerebro = bt.Cerebro()
    data = SignalData(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(SignalStrategy)
    cerebro.broker.setcash(START_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    start_value = cerebro.broker.getvalue()
    cerebro.run()
    end_value = cerebro.broker.getvalue()

    pnl = end_value - start_value
    logging.info("Start value: %.2f", start_value)
    logging.info("End value:   %.2f", end_value)
    logging.info("PnL:         %.2f", pnl)
    logging.info("Signals: BUY=%s SELL=%s TOTAL=%s", buy_signals, sell_signals, total_signals)

    if len(df.index) > 1:
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            per_day = total_signals / days
            logging.info("Signals per day: %.2f", per_day)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
