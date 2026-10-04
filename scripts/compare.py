"""Is variant B really better than variant A, or is it seed luck?

Usage:  python scripts/compare.py baseline small_net [--symbol BTC-USD] [--period val]
        python scripts/compare.py baseline:last baseline:best     # same run, two selection rules

Each run is "name" or "name:rule", where rule is last (default), smoothed or best.

For each metric: the IQM of each run, the difference (B minus A) with a 95%
bootstrap interval, and how often a B seed beats an A seed. "clear" means the
interval excludes zero, i.e. the difference is bigger than the seed noise.
"""
import argparse
from pathlib import Path

import pandas as pd

from rltrader.stats import bootstrap_ci, compare, iqm

p = argparse.ArgumentParser()
p.add_argument("a")
p.add_argument("b")
p.add_argument("--symbol", default="BTC-USD")
p.add_argument("--period", default="val", choices=["train", "val", "test"])
args = p.parse_args()

root = Path("results") / args.symbol


def read(spec: str) -> pd.DataFrame:
    name, _, rule = spec.partition(":")
    table = pd.read_csv(root / name / f"{args.period}_metrics.csv")
    return table[table.selection == (rule or "last")]


a, b = read(args.a), read(args.b)

fmt = lambda x: "{:.3f} [{:.3f}, {:.3f}]".format(iqm(x), *bootstrap_ci(x))
rows = []
for metric in ["log_return_per_year", "sharpe", "max_drawdown", "turnover"]:
    c = compare(a[metric], b[metric])
    rows.append({
        "metric": metric,
        f"A = {args.a}": fmt(a[metric]),
        f"B = {args.b}": fmt(b[metric]),
        "B - A": "{:+.3f} [{:+.3f}, {:+.3f}]".format(c["difference"], c["ci_low"], c["ci_high"]),
        "P(B > A)": f"{c['prob_improvement']:.0%}",
        "clear": "yes" if c["clear"] else "no",
    })
print(f"{args.period} period, {len(a)} vs {len(b)} seeds (IQM [95% interval])")
print(pd.DataFrame(rows).set_index("metric").to_string())
