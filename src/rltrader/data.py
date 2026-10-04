"""Loading price data and splitting it by time.

Fixes from the 2023 notebook:
  * the data is always sorted oldest -> newest (the old CSV was newest-first
    and never sorted, so the agent traded backwards in time);
  * the split is by calendar date, so the test period is always strictly
    *after* everything the agent trained on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def download(symbol: str, path: Path, start: str = "2014-01-01") -> pd.DataFrame:
    """Download daily bars from Yahoo Finance and save them as a CSV."""
    import yfinance as yf

    raw = yf.download(symbol, start=start, interval="1d", auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"No data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):  # yfinance returns (field, ticker) columns
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[COLUMNS]
    df.index.name = "date"
    # today's bar is still forming, so its "close" is not final yet
    df = df[df.index < pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return load(path)


def load(path: Path) -> pd.DataFrame:
    """Load a CSV of bars, guaranteeing a clean, ascending time index."""
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df[COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    assert df.index.is_monotonic_increasing
    return df


@dataclass(frozen=True)
class Split:
    """Chronological train / validation / test periods.

    train: the agent learns here.
    val:   used to pick the best checkpoint (model selection).
    test:  touched once, at the very end, to report results.
    """

    train_end: str = "2021-12-31"
    val_end: str = "2023-12-31"

    def periods(self, df: pd.DataFrame) -> dict[str, tuple[str | None, str | None]]:
        """(start, end) dates of each period, to pass to ``TradingEnv``."""
        day = pd.Timedelta(days=1)
        train_end, val_end = pd.Timestamp(self.train_end), pd.Timestamp(self.val_end)
        return {
            "train": (None, str(train_end.date())),
            "val": (str((train_end + day).date()), str(val_end.date())),
            "test": (str((val_end + day).date()), str(df.index[-1].date())),
        }
