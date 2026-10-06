"""Score a finished run on a period, across all its evaluation assets.

Usage:  python scripts/evaluate_test.py rw_vol_scaled [--period test] [--symbol BTC-USD]

Loads the saved final models ("last" rule) of every seed and runs them on
each evaluation asset, next to three references under the same trading rules:
buy-and-hold, the SMA 20/50 rule, and the average of 100 random agents.

The test period is the final exam. Run this once per phase, for the setup
already chosen on validation, and never use the answer to choose settings.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from rltrader.dqn import DQNAgent, DQNConfig
from rltrader.evaluate import (
    agent_policy, buy_and_hold, metrics, random_policy, run_policy, sma_crossover,
)
from rltrader.experiment import RunConfig, make_env
from rltrader.stats import bootstrap_ci, iqm

N_RANDOM = 100
KEY = "log_return_per_year"

p = argparse.ArgumentParser()
p.add_argument("name")
p.add_argument("--symbol", default="BTC-USD")
p.add_argument("--period", default="test", choices=["val", "test"])
args = p.parse_args()

run_dir = Path("results") / args.symbol / args.name
raw = json.loads((run_dir / "config.json").read_text())
n_seeds = raw.pop("seeds")
raw["train_symbols"], raw["eval_symbols"] = tuple(raw["train_symbols"]), tuple(raw["eval_symbols"])
cfg = RunConfig(**raw)
dqn_cfg = DQNConfig(**cfg.dqn)

rows, reference = [], {}
for s in cfg.eval_symbols:
    env = make_env(cfg, args.period, symbol=s)
    bh = metrics(run_policy(env, buy_and_hold))
    sma = metrics(run_policy(env, sma_crossover()))
    rnd = iqm([metrics(run_policy(env, random_policy(i)))[KEY] for i in range(N_RANDOM)])
    reference[s] = {"random": rnd, "sma": sma[KEY], "buy_hold": bh[KEY],
                    "sma_trades": sma["turnover"], "buy_hold_drawdown": bh["max_drawdown"],
                    "sma_drawdown": sma["max_drawdown"]}
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, dqn_cfg)
    for seed in range(n_seeds):
        agent.load(run_dir / "checkpoints" / f"seed{seed}_last.pt")
        m = metrics(run_policy(env, agent_policy(agent)))
        rows.append({"seed": seed, "symbol": s, **m,
                     "edge_vs_random": m[KEY] - rnd,
                     "edge_vs_sma": m[KEY] - sma[KEY],
                     "edge_vs_buy_hold": m[KEY] - bh[KEY]})

long = pd.DataFrame(rows)
long.to_csv(run_dir / f"{args.period}_assets.csv", index=False)
ref = pd.DataFrame(reference).T

table = long.groupby("symbol", sort=False).agg(
    dqn=(KEY, iqm), dqn_trades=("turnover", iqm), dqn_drawdown=("max_drawdown", iqm))
table = table.join(ref)[["dqn", "random", "sma", "buy_hold", "dqn_trades", "sma_trades",
                         "dqn_drawdown", "sma_drawdown", "buy_hold_drawdown"]]
table.loc["MEAN over assets"] = table.mean()
start, end = env.dates[env.first_t].date(), env.dates[env.last_t].date()
print(f"{args.name}: {args.period} period (last asset: {start} to {end}), "
      f"{n_seeds} seeds, log return per year")
print(table.round(3).to_string())

per_seed = long.drop(columns="symbol").groupby("seed").mean()
print("\nAveraged over assets, IQM over seeds [95% interval]:")
for col in ["edge_vs_random", "edge_vs_sma", "edge_vs_buy_hold", KEY, "sharpe", "max_drawdown"]:
    lo, hi = bootstrap_ci(per_seed[col])
    print(f"  {col:22s} {iqm(per_seed[col]):+.3f} [{lo:+.3f}, {hi:+.3f}]")
print(f"  seeds with positive edge over random: {(per_seed.edge_vs_random > 0).sum()} of {n_seeds}")
