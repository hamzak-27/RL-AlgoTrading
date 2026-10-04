"""Train DQN agents and compare them with baselines on unseen data.

Usage:  python scripts/train.py --symbol BTC-USD --seeds 5 --steps 60000

For each seed:
  1. train on the training period, in random one-year episodes;
  2. every few thousand steps, test the greedy policy on the validation
     period and keep the best checkpoint;
  3. run that checkpoint once on the test period.
Baselines run through the same environment with the same fees.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rltrader.data import Split, load
from rltrader.dqn import DQNAgent, DQNConfig
from rltrader.env import TradingEnv
from rltrader.evaluate import (
    agent_policy, buy_and_hold, metrics, random_policy, run_policy, sma_crossover,
)

p = argparse.ArgumentParser()
p.add_argument("--symbol", default="BTC-USD")
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--steps", type=int, default=60_000)
p.add_argument("--eval-every", type=int, default=2_000)
p.add_argument("--window", type=int, default=10)
p.add_argument("--fee", type=float, default=0.001)
p.add_argument("--episode-length", type=int, default=365)
p.add_argument("--out", default="results")
args = p.parse_args()

out = Path(args.out) / args.symbol
out.mkdir(parents=True, exist_ok=True)

df = load(Path("data") / f"{args.symbol}.csv")
periods = Split().periods(df)


def make_env(period: str, episode_length=None) -> TradingEnv:
    start, end = periods[period]
    return TradingEnv(df, window=args.window, fee=args.fee,
                      episode_length=episode_length, start=start, end=end)


def train_one(seed: int):
    train_env, val_env = make_env("train", args.episode_length), make_env("val")
    cfg = DQNConfig(eps_decay_steps=args.steps // 2)
    agent = DQNAgent(train_env.observation_space.shape[0], train_env.action_space.n, cfg, seed)
    ckpt = out / f"dqn_seed{seed}.pt"
    best_val, curve = -np.inf, []

    obs, _ = train_env.reset(seed=seed)
    for step in range(1, args.steps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = train_env.reset()

        if step % args.eval_every == 0:
            val_log_return = float(np.log(run_policy(val_env, agent_policy(agent))["value"].iloc[-1]))
            curve.append({"seed": seed, "step": step, "val_log_return": val_log_return,
                          "epsilon": agent.epsilon})
            if val_log_return > best_val:
                best_val = val_log_return
                agent.save(ckpt)

    agent.load(ckpt)
    print(f"seed {seed}: best validation log-return {best_val:+.3f}")
    return agent, curve


# ------------------------------------------------------------------- run it
records: dict[str, pd.DataFrame] = {}
rows, curves = [], []

test_env = make_env("test")
for name, policy in [
    ("Buy & hold", buy_and_hold),
    ("SMA 20/50", sma_crossover()),
    ("Random", random_policy(0)),
]:
    records[name] = run_policy(test_env, policy)
    rows.append({"strategy": name, "seed": "-", **metrics(records[name])})

for seed in range(args.seeds):
    agent, curve = train_one(seed)
    curves += curve
    records[f"DQN seed {seed}"] = run_policy(test_env, agent_policy(agent))
    rows.append({"strategy": "DQN", "seed": seed, **metrics(records[f"DQN seed {seed}"])})

table = pd.DataFrame(rows)
table.to_csv(out / "test_metrics.csv", index=False)
pd.DataFrame(curves).to_csv(out / "validation_curves.csv", index=False)
pd.DataFrame({k: v["value"] for k, v in records.items()}).to_csv(out / "test_equity.csv")
(out / "config.json").write_text(json.dumps({**vars(args), "periods": periods}, indent=2))

numeric = table.drop(columns=["seed"]).groupby("strategy", sort=False)
summary = numeric.mean().round(3)
summary["sharpe_std"] = numeric["sharpe"].std().round(3)
print(f"\nTest period {periods['test'][0]} -> {periods['test'][1]}, fee {args.fee:.2%} per trade")
print(summary.to_string())
print("\nPer-seed DQN:")
print(table[table.strategy == "DQN"].drop(columns="strategy").round(3).to_string(index=False))

# --------------------------------------------------------------------- chart
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
dqn = pd.DataFrame({k: v["value"] for k, v in records.items() if k.startswith("DQN")})
for col in dqn:  # individual seeds, faint
    ax.plot(dqn.index, dqn[col], color=BLUE, lw=0.8, alpha=0.3)
series = [
    (f"DQN (mean of {args.seeds} seeds)", dqn.mean(axis=1), BLUE),
    ("Buy & hold", records["Buy & hold"]["value"], ORANGE),
    ("SMA 20/50", records["SMA 20/50"]["value"], AQUA),
]
for label, s, color in series:
    ax.plot(s.index, s, color=color, lw=2, label=label)
ax.axhline(1.0, color="#c3c2b7", lw=1)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(axis="y", color=GRID, lw=0.8)
ax.tick_params(colors=MUTED, length=0)
for side in ax.spines.values():
    side.set_visible(False)
ax.set_title(f"{args.symbol}: portfolio value on the test period (start = 1×, "
             f"{args.fee:.1%} fee per trade)", color=INK, loc="left", fontsize=12)
ax.legend(frameon=False, labelcolor="#52514e", loc="upper left")
fig.tight_layout()
fig.savefig(out / "test_equity.png", dpi=150)
print(f"\nSaved results to {out}")
