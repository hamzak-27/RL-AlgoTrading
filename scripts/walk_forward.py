"""Walk-forward test of the final setup: retrain each year, test on the next.

Usage:
    python scripts/walk_forward.py --method dqn --name wf_dqn
    python scripts/walk_forward.py --method gbm --name wf_gbm
    python scripts/walk_forward.py --compare wf_dqn wf_gbm

Defaults are the Phase B final setup (11 assets, 10-day hold, 3-day window,
volatility-scaled reward). Every (year, seed) pair is one job, run in parallel.
Results go to results/walk_forward/<name>/. Each year is saved as it finishes;
running the same command again continues from where it stopped.
"""
import sys

if "gbm" in sys.argv:
    # On Windows LightGBM must be loaded before PyTorch, or its first fit crashes
    # (the two bring different copies of the OpenMP runtime).
    import lightgbm  # noqa: F401

import argparse
import ast
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from rltrader.evaluate import buy_and_hold, metrics, random_policy, run_policy, sma_crossover
from rltrader.experiment import RunConfig
from rltrader.stats import bootstrap_ci, compare, iqm
from rltrader.walkforward import METHODS, fold_envs, make_folds

ALL = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "LTC-USD", "ADA-USD",
       "DOGE-USD", "SPY", "QQQ", "GLD"]
KEY = "log_return_per_year"
N_RANDOM = 100
ROOT = Path("results") / "walk_forward"
EDGES = ["edge_vs_random", "edge_vs_sma", "edge_vs_buy_hold"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two finished runs")
    p.add_argument("--method", choices=list(METHODS), default="dqn")
    p.add_argument("--name")
    p.add_argument("--first-year", type=int, default=2019)
    p.add_argument("--last-year", type=int, default=2026)
    p.add_argument("--symbols", nargs="*", default=ALL)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--min-hold", type=int, default=10)
    p.add_argument("--reward", default="vol_scaled")
    p.add_argument("--dqn", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def per_seed(long: pd.DataFrame, by=("seed",)) -> pd.DataFrame:
    """Average over assets (and years, unless ``by`` includes fold) for each seed."""
    return long.groupby(list(by))[EDGES + [KEY, "sharpe", "max_drawdown", "turnover"]].mean()


def interval(x) -> str:
    return "{:+.3f} [{:+.3f}, {:+.3f}]".format(iqm(x), *bootstrap_ci(x))


def report(long: pd.DataFrame, ref: pd.DataFrame, name: str) -> None:
    rows = {}
    for fold, g in long.groupby("fold"):
        s = per_seed(g)
        r = ref[ref.fold == fold]
        rows[fold] = {"assets": g.symbol.nunique(), "model": iqm(s[KEY]),
                      "random": r["random"].mean(), "sma": r["sma"].mean(),
                      "buy_hold": r["buy_hold"].mean(),
                      "edge_vs_random": interval(s["edge_vs_random"]),
                      "edge_vs_sma": interval(s["edge_vs_sma"]),
                      "trades": iqm(s["turnover"])}
    table = pd.DataFrame(rows).T[["assets", "model", "random", "sma", "buy_hold",
                                  "edge_vs_random", "edge_vs_sma", "trades"]]
    for col in ["model", "random", "sma", "buy_hold", "trades"]:
        table[col] = table[col].astype(float).round(3)
    print(f"{name}: log return per year, averaged over assets; IQM over seeds [95% interval]")
    print(table.to_string())

    overall = per_seed(long)
    print("\nAll years and assets together:")
    for col in EDGES + [KEY, "sharpe", "max_drawdown"]:
        print(f"  {col:22s} {interval(overall[col])}")
    by_fold = per_seed(long, by=("fold", "seed")).groupby("fold")
    wins_random = (by_fold["edge_vs_random"].apply(iqm) > 0).sum()
    wins_sma = (by_fold["edge_vs_sma"].apply(iqm) > 0).sum()
    print(f"  years ahead of random: {wins_random} of {len(by_fold)};"
          f" years ahead of SMA: {wins_sma} of {len(by_fold)}")


def run(args) -> None:
    out = ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = RunConfig(symbol=args.symbols[0], train_symbols=tuple(args.symbols),
                    eval_symbols=tuple(args.symbols), steps=args.steps, window=args.window,
                    fee=args.fee, min_hold=args.min_hold, reward_type=args.reward,
                    dqn={k: ast.literal_eval(v) for k, v in (kv.split("=", 1) for kv in args.dqn)})
    folds = make_folds(args.first_year, args.last_year)
    (out / "config.json").write_text(json.dumps(
        {**cfg.to_dict(), "method": args.method, "seeds": args.seeds,
         "folds": [f.__dict__ for f in folds]}, indent=2))

    # References under the same rules, for every (year, asset).
    if (out / "reference.csv").exists():
        ref = pd.read_csv(out / "reference.csv", dtype={"fold": str})
    else:
        ref_rows = []
        for fold in folds:
            for s, env in fold_envs(cfg, fold, args.symbols, "test").items():
                rnd = iqm([metrics(run_policy(env, random_policy(i)))[KEY]
                           for i in range(N_RANDOM)])
                ref_rows.append({"fold": fold.name, "symbol": s, "random": rnd,
                                 "sma": metrics(run_policy(env, sma_crossover()))[KEY],
                                 "buy_hold": metrics(run_policy(env, buy_and_hold))[KEY]})
        ref = pd.DataFrame(ref_rows)
        ref.to_csv(out / "reference.csv", index=False)

    # Each year is saved as soon as it finishes, and a rerun skips the years
    # already on disk, so an interrupted run loses at most one year of work.
    def fold_file(fold):
        return out / f"year_{fold.name}.csv"

    todo = [f for f in folds if not fold_file(f).exists()]
    for fold in folds:
        if fold not in todo:
            print(f"year {fold.name} already done, skipping", flush=True)

    if args.method == "gbm":
        # Each fit takes about a second, so no worker processes are needed.
        for fold in todo:
            rows = [r for seed in range(args.seeds) for r in METHODS["gbm"](cfg, seed, fold)]
            pd.DataFrame(rows).to_csv(fold_file(fold), index=False)
            print(f"year {fold.name} done", flush=True)
    elif todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {fold.name: [pool.submit(METHODS[args.method], cfg, seed, fold)
                                for seed in range(args.seeds)] for fold in todo}
            for fold in todo:
                rows = [r for job in jobs[fold.name] for r in job.result()]
                pd.DataFrame(rows).to_csv(fold_file(fold), index=False)
                print(f"year {fold.name} done", flush=True)

    results = pd.concat([pd.read_csv(fold_file(f), dtype={"fold": str}) for f in folds])
    long = results.merge(ref, on=["fold", "symbol"])
    for col, base in zip(EDGES, ["random", "sma", "buy_hold"]):
        long[col] = long[KEY] - long[base]
    long.to_csv(out / "folds.csv", index=False)
    print()
    report(long, ref, args.name)
    print(f"\nSaved to {out}")


def compare_runs(a: str, b: str) -> None:
    la, lb = (pd.read_csv(ROOT / n / "folds.csv", dtype={"fold": str}) for n in (a, b))
    sa, sb = per_seed(la), per_seed(lb)
    rows = []
    for col in EDGES + ["sharpe", "max_drawdown", "turnover"]:
        c = compare(sa[col], sb[col])
        rows.append({"metric": col, f"A = {a}": interval(sa[col]), f"B = {b}": interval(sb[col]),
                     "B - A": "{:+.3f} [{:+.3f}, {:+.3f}]".format(
                         c["difference"], c["ci_low"], c["ci_high"]),
                     "P(B > A)": f"{c['prob_improvement']:.0%}",
                     "clear": "yes" if c["clear"] else "no"})
    print("All years and assets, IQM over seeds [95% interval]")
    print(pd.DataFrame(rows).set_index("metric").to_string())


if __name__ == "__main__":  # required on Windows for parallel workers
    args = parse_args()
    if args.compare:
        compare_runs(*args.compare)
    else:
        if not args.name:
            raise SystemExit("--name is required")
        run(args)
