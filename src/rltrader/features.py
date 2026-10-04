"""Turning raw prices into inputs a neural network can learn from.

The 2023 notebook fed the network sigmoid(price_today - price_yesterday).
Bitcoin moves hundreds of dollars a day, so that was ~always exactly 0 or 1.

Rules followed here:
  1. Use *returns* (percentage moves), not dollar differences, so a move in
     2015 at $300 and a move in 2024 at $60,000 look alike.
  2. Scale every feature to roughly [-3, 3] so no single one dominates.
  3. Every rolling statistic looks only backwards. The value in row t uses
     bars up to and including t, never later ones (no look-ahead leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOL_WINDOW = 30
WARMUP = 60  # rows dropped at the start, where rolling windows are not yet full


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-12))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature table aligned with ``df`` (first WARMUP rows removed)."""
    close = df["close"]
    logret = np.log(close).diff()
    vol = logret.rolling(VOL_WINDOW).std()  # recent daily volatility

    f = pd.DataFrame(index=df.index)
    # Returns over 1, 5 and 20 days, each divided by how volatile the market
    # has been lately ("how unusual was this move?").
    for k in (1, 5, 20):
        f[f"ret_{k}"] = np.log(close).diff(k) / (vol * np.sqrt(k))
    # Is volatility high or low compared with the last two months?
    f["vol_ratio"] = np.log(vol / vol.rolling(WARMUP).mean())
    # Distance of price from its 50-day average, in volatility units (trend).
    sma = close.rolling(50).mean()
    f["trend"] = np.log(close / sma) / (vol * np.sqrt(50))
    # RSI rescaled from [0, 100] to [-1, 1] (overbought / oversold).
    f["rsi"] = (_rsi(close) - 50) / 50
    # Is today's volume unusual versus the last two months?
    logv = np.log(df["volume"].replace(0, np.nan)).ffill()
    f["volume_z"] = (logv - logv.rolling(WARMUP).mean()) / (logv.rolling(WARMUP).std() + 1e-12)
    # Intraday range relative to recent volatility.
    f["range"] = np.log(df["high"] / df["low"]) / vol - 1.0

    f = f.clip(-5, 5)
    return f.iloc[WARMUP + VOL_WINDOW:].dropna()
