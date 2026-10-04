"""Download daily price history.  Usage: python scripts/download_data.py BTC-USD"""
import sys
from pathlib import Path

from rltrader.data import download

symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
path = Path("data") / f"{symbol}.csv"
df = download(symbol, path)
print(f"{symbol}: {len(df)} daily bars, {df.index[0].date()} -> {df.index[-1].date()}, saved to {path}")
