"""Train DQN agents over several seeds and compare them with baselines.

Usage:
    python scripts/train.py --name baseline --seeds 10
    python scripts/train.py --name small_net --seeds 10 --dqn hidden=32 lr=1e-4

For each seed (run in parallel):
  1. train on the training period, in random one-year episodes;
  2. every few thousand steps, test the greedy policy on the validation period;
  3. report three ways of choosing the checkpoint: "last" (the final model, no
     choosing, so its validation score is unbiased), "smoothed" (end of the
     best run of 5 evaluations) and "best" (single highest score, flattering).
Baselines run through the same environment with the same fees.

Results go to results/<symbol>/<name>/. Validation numbers are printed;
test numbers are saved but only printed with --show-test, so that day-to-day
decisions are not made by peeking at the test period.
"""
import argparse
import ast
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from rltrader.evaluate import buy_and_hold, metrics, random_policy, run_policy, sma_crossover
from rltrader.experiment import RunConfig, make_env, train_seed
from rltrader.stats import bootstrap_ci, iqm

SHOWN = ["log_return", "sharpe", "max_drawdown", "exposure", "turnover"]
N_RANDOM = 100
RULES = ["last", "smoothed", "best"]  # checkpoint-selection rules; "last" is the headline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="label for this variant, e.g. baseline")
    p.add_argument("--symbol", default="BTC-USD",
                   help="main asset: checkpoints and headline numbers")
    p.add_argument("--train-symbols", nargs="*", default=[],
                   help="assets to learn from (default: just --symbol)")
    p.add_argument("--eval-symbols", nargs="*", default=[],
                   help="assets to also score the final agent on (validation period)")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--fee", type=float, default=0.001)
    p.add_argument("--episode-length", type=int, default=365)
    p.add_argument("--reward", default="log_return",
                   choices=["log_return", "vol_scaled", "dsr", "drawdown", "downside"],
                   help="what the agent is taught to maximise")
    p.add_argument("--reward-param", type=float, default=1.0,
                   help="strength of the drawdown / downside penalty")
    p.add_argument("--switch-penalty", type=float, default=0.0,
                   help="reward penalty per unit of position changed")
    p.add_argument("--min-hold", type=int, default=0,
                   help="minimum days a position must be held after a trade")
    p.add_argument("--dqn", nargs="*", default=[], metavar="KEY=VALUE",
                   help="DQNConfig overrides, e.g. hidden=32 lr=1e-4")
    p.add_argument("--show-test", action="store_true")
    return p.parse_args()


def summarise(dqn: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy; DQN shown as IQM with a 95% confidence interval."""
    rows = {name: {m: f"{row[m]:.3f}" for m in SHOWN} for name, row in baselines.iterrows()}
    for rule in RULES:
        d = dqn[dqn.selection == rule]
        rows[f"DQN {rule} (IQM of {len(d)} seeds)"] = {m: f"{iqm(d[m]):.3f}" for m in SHOWN}
        rows[f"DQN {rule} 95% interval"] = {
            m: "[{:.3f}, {:.3f}]".format(*bootstrap_ci(d[m])) for m in SHOWN
        }
    return pd.DataFrame(rows).T


def report_assets(cfg, results, out: Path) -> pd.DataFrame:
    """Score the final agents on every evaluation asset and save two files:
    one row per (seed, asset), and one row per seed averaged over assets."""
    reference = {}
    for s in cfg.eval_symbols:
        env = make_env(cfg, "val", symbol=s)
        bh = metrics(run_policy(env, buy_and_hold))["log_return_per_year"]
        rnd = iqm([metrics(run_policy(env, random_policy(i)))["log_return_per_year"]
                   for i in range(N_RANDOM)])
        reference[s] = {"buy_hold": bh, "random": rnd}

    rows = []
    for r in results:
        for s, m in r["val_assets"].items():
            rows.append({"seed": r["seed"], "symbol": s, **m,
                         # edge = how far ahead of a no-skill agent under the same rules
                         "edge_vs_random": m["log_return_per_year"] - reference[s]["random"],
                         "edge_vs_buy_hold": m["log_return_per_year"] - reference[s]["buy_hold"]})
    long = pd.DataFrame(rows)
    long.to_csv(out / "val_assets.csv", index=False)

    per_seed = long.drop(columns="symbol").groupby("seed").mean().reset_index()
    per_seed.insert(1, "selection", "last")
    per_seed.to_csv(out / "val_multi_metrics.csv", index=False)

    table = long.groupby("symbol", sort=False).agg(
        dqn=("log_return_per_year", iqm), edge_vs_random=("edge_vs_random", iqm),
        trades=("turnover", iqm))
    table.insert(1, "random", [reference[s]["random"] for s in table.index])
    table.insert(2, "buy_hold", [reference[s]["buy_hold"] for s in table.index])
    lo, hi = bootstrap_ci(per_seed["edge_vs_random"])
    table.loc["MEAN over assets"] = table.mean()
    print(f"\nMean edge over random across assets: {iqm(per_seed['edge_vs_random']):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] (IQM over seeds, 95% interval)")
    return table


def plot_equity(equity: pd.DataFrame, baselines: dict, title: str, path: Path) -> None:
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for col in equity:  # individual seeds, faint
        ax.plot(equity.index, equity[col], color=BLUE, lw=0.8, alpha=0.3)
    ax.plot(equity.index, equity.mean(axis=1), color=BLUE, lw=2,
            label=f"DQN (mean of {equity.shape[1]} seeds)")
    for (label, s), color in zip(baselines.items(), (ORANGE, AQUA)):
        ax.plot(s.index, s, color=color, lw=2, label=label)
    ax.axhline(1.0, color="#c3c2b7", lw=1)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, length=0, which="both")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, labelcolor="#52514e", loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    dqn_overrides = {k: ast.literal_eval(v) for k, v in (kv.split("=", 1) for kv in args.dqn)}
    cfg = RunConfig(symbol=args.symbol, train_symbols=tuple(args.train_symbols),
                    eval_symbols=tuple(args.eval_symbols), steps=args.steps, eval_every=args.eval_every,
                    window=args.window, fee=args.fee, episode_length=args.episode_length,
                    switch_penalty=args.switch_penalty, min_hold=args.min_hold,
                    reward_type=args.reward, reward_param=args.reward_param,
                    dqn=dqn_overrides)
    out = Path("results") / args.symbol / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**cfg.to_dict(), "seeds": args.seeds}, indent=2))

    # ---- baselines, on every period
    baseline_tables, baseline_equity = {}, {}
    for period in ("train", "val", "test"):
        policies = {"Buy & hold": buy_and_hold, "SMA 20/50": sma_crossover()}
        env = make_env(cfg, period)
        records = {name: run_policy(env, pol) for name, pol in policies.items()}
        rows = {n: metrics(r) for n, r in records.items()}
        # One random agent is as noisy as one DQN seed, so use many: this row is
        # "what you get from the trading rules alone, with no skill at all".
        randoms = pd.DataFrame([metrics(run_policy(env, random_policy(s)))
                                for s in range(N_RANDOM)])
        rows[f"Random (IQM of {N_RANDOM})"] = {m: iqm(randoms[m]) for m in randoms}
        baseline_tables[period] = pd.DataFrame(rows).T
        baseline_tables[period].to_csv(out / f"baselines_{period}.csv")
        baseline_equity[period] = {n: records[n]["value"] for n in ("Buy & hold", "SMA 20/50")}

    # ---- train all seeds in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(train_seed, cfg, seed, str(out)) for seed in range(args.seeds)]
        results = []
        for f in futures:
            r = f.result()
            results.append(r)
            print(f"seed {r['seed']} done: validation log-return "
                  + ", ".join(f"{rule} {r['val'][rule]['log_return']:+.3f}" for rule in RULES),
                  flush=True)

    tables = {}
    for period in ("train", "val", "test"):
        tables[period] = pd.DataFrame([{"seed": r["seed"], "selection": rule, **r[period][rule]}
                                       for r in results for rule in RULES])
        tables[period].to_csv(out / f"{period}_metrics.csv", index=False)
    pd.DataFrame([row for r in results for row in r["curve"]]).to_csv(
        out / "validation_curves.csv", index=False)
    equity = pd.DataFrame({f"seed {r['seed']}": r["test_equity"]["last"] for r in results})
    equity.to_csv(out / "test_equity.csv")
    plot_equity(equity, baseline_equity["test"],
                f"{args.symbol} / {args.name}: portfolio value on the test period "
                f"(start = 1×, {args.fee:.1%} fee per trade)", out / "test_equity.png")

    for period in ("val", "test") if args.show_test else ("val",):
        print(f"\n{period.upper()} period, fee {args.fee:.2%} per trade")
        print(summarise(tables[period], baseline_tables[period]).to_string())
    if cfg.eval_symbols:
        assets = report_assets(cfg, results, out)
        print("\nVALIDATION period on every evaluation asset (last rule, log return per year)")
        print(assets.round(3).to_string())

    # Overfitting check: how far ahead of buy-and-hold is the agent on the data it
    # learned from, versus on data it has never seen? Measured against buy-and-hold
    # because the training years were a much stronger market than the validation years.
    edge = {}
    for period in ("train", "val"):
        dqn = iqm(tables[period][tables[period].selection == "last"]["log_return_per_year"])
        edge[period] = dqn - baseline_tables[period].loc["Buy & hold", "log_return_per_year"]
    print("\nOverfitting check (last rule, log return per year vs buy-and-hold): "
          f"train {edge['train']:+.3f}, validation {edge['val']:+.3f}, "
          f"gap {edge['train'] - edge['val']:.3f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":  # required on Windows for parallel workers
    main()
