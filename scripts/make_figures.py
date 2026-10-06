"""Build the README figures and summary tables from saved results.

Usage:  python scripts/make_figures.py

Reads results/ (which is not committed) and writes docs/figures/*.png and
docs/results/*.csv (which are), so the numbers behind every chart are in the repo.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

from rltrader.evaluate import metrics, random_policy, run_policy, sma_crossover
from rltrader.experiment import RunConfig, make_env
from rltrader.stats import bootstrap_ci, iqm

RESULTS, FIGS, TABLES = Path("results"), Path("docs/figures"), Path("docs/results")
KEY = "log_return_per_year"

# Colours: one fixed colour per strategy, everywhere.
SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # DQN, LightGBM, SMA


def new_axes(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    return fig, ax


def titles(fig, ax, title: str, subtitle: str) -> None:
    """Title and subtitle at the top-left of the figure; the plot starts below them."""
    height = fig.get_figheight()
    lines = subtitle.count("\n") + 1
    fig.text(0.012, 1 - 0.12 / height, title, ha="left", va="top", fontsize=13,
             color=INK, fontweight="bold")
    fig.text(0.012, 1 - 0.46 / height, subtitle, ha="left", va="top", fontsize=10,
             color=SECONDARY, linespacing=1.35)
    fig._plot_top = 1 - (0.62 + 0.21 * lines) / height


def save(fig, name: str) -> None:
    fig.tight_layout(rect=(0, 0, 1, fig._plot_top))
    fig.savefig(FIGS / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", FIGS / name)


def seed_summary(values) -> tuple[float, float, float]:
    lo, hi = bootstrap_ci(values)
    return iqm(values), lo, hi


# --------------------------------------------------------- 1. walk-forward
def walk_forward() -> None:
    ref = pd.read_csv(RESULTS / "walk_forward/wf_dqn/reference.csv", dtype={"fold": str})
    runs = {"DQN": "wf_dqn", "LightGBM": "wf_gbm"}
    rows = []
    for year, r in ref.groupby("fold"):
        row = {"year": year, "assets": len(r), "random": r["random"].mean(),
               "sma": r["sma"].mean(), "buy_hold": r["buy_hold"].mean()}
        row["sma_edge_vs_random"] = row["sma"] - row["random"]
        for label, name in runs.items():
            long = pd.read_csv(RESULTS / f"walk_forward/{name}/folds.csv", dtype={"fold": str})
            per_seed = long[long.fold == year].groupby("seed")[[KEY, "edge_vs_random"]].mean()
            key = label.lower()
            row[key] = iqm(per_seed[KEY])
            row[f"{key}_edge_vs_random"], row[f"{key}_edge_lo"], row[f"{key}_edge_hi"] = (
                seed_summary(per_seed["edge_vs_random"]))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.round(4).to_csv(TABLES / "walk_forward_by_year.csv", index=False)

    fig, ax = new_axes(10.5, 5.2)
    x = range(len(table))
    width = 0.24
    series = [("DQN agent", "dqn", BLUE), ("LightGBM (supervised)", "lightgbm", ORANGE),
              ("SMA 20/50 rule", "sma", AQUA)]
    for i, (label, key, color) in enumerate(series):
        pos = [v + (i - 1) * (width + 0.03) for v in x]
        heights = table[f"{key}_edge_vs_random"]
        ax.bar(pos, heights, width, color=color, label=label, zorder=3)
        if f"{key}_edge_lo" in table:  # 95% interval over seeds
            ax.vlines(pos, table[f"{key}_edge_lo"], table[f"{key}_edge_hi"],
                      color=INK, lw=1, zorder=4)
    ax.axhline(0, color=AXIS, lw=1, zorder=2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_xticks(list(x), table["year"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}"))
    ax.legend(frameon=False, labelcolor=SECONDARY, loc="upper right", fontsize=10)
    titles(fig, ax, "Every method beats random trading in most years; none stands out",
           "Edge over the average of 100 random agents, log return per year. Each year is traded "
           "by models trained only on earlier years.\nUp to 11 assets, 0.1% fee, 10-day minimum "
           "hold. Black lines: 95% interval over 10 seeds. 2026 runs to early October.")
    save(fig, "walk_forward_edge.png")


# ------------------------------------------------------- 2. selection bias
def selection_bias() -> None:
    val = pd.read_csv(RESULTS / "BTC-USD/baseline/val_metrics.csv")
    env = make_env(RunConfig(), "val")
    random_ref = iqm([metrics(run_policy(env, random_policy(i)))["log_return"]
                      for i in range(100)])
    buy_hold = pd.read_csv(RESULTS / "BTC-USD/baseline/baselines_val.csv",
                           index_col=0).loc["Buy & hold", "log_return"]
    rules = [("last", "Final model\n(no choosing)"),
             ("smoothed", "Best run of\n5 evaluations"),
             ("best", "Single best\nevaluation")]
    rows = []
    for rule, label in rules:
        mid, lo, hi = seed_summary(val[val.selection == rule]["log_return"])
        rows.append({"rule": rule, "label": label, "log_return": mid, "lo": lo, "hi": hi})
    table = pd.DataFrame(rows)
    table.assign(buy_hold=buy_hold, random=random_ref).drop(columns="label").round(4).to_csv(
        TABLES / "selection_bias.csv", index=False)

    fig, ax = new_axes(10.5, 3.9)
    y = list(range(len(table)))[::-1]
    ax.hlines(y, table["lo"], table["hi"], color=BLUE, lw=2, zorder=3)
    ax.scatter(table["log_return"], y, s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
    for yy, v in zip(y, table["log_return"]):
        ax.text(v, yy + 0.22, f"{v:+.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    for value, label in [(random_ref, "random agents"), (buy_hold, "buy & hold")]:
        ax.axvline(value, color=AXIS, lw=1, zorder=2)
        ax.text(value + 0.008, len(y) - 0.45, f"{label} {value:+.2f}", ha="left", va="bottom",
                fontsize=9, color=MUTED)
    ax.set_yticks(y, table["label"])
    ax.tick_params(axis="y", labelcolor=SECONDARY)
    ax.set_ylim(-0.6, len(y) - 0.25)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.1f}"))
    titles(fig, ax, "The same ten agents, three ways of choosing which saved copy to report",
           "Validation log return on Bitcoin, 2022-2023, for the first rebuilt DQN before any "
           "tuning.\nDot: interquartile mean of 10 seeds; line: 95% interval.")
    save(fig, "selection_bias.png")


# ------------------------------------------------------------ 3. ablations
ABLATIONS = [
    ("Training data", [
        ("window3", "Bitcoin only"),
        ("multi_crypto", "8 crypto assets"),
        ("multi_all", "8 crypto + SPY, QQQ, GLD  (kept)"),
        ("multi_crypto_long", "8 crypto, 2.5x longer training"),
    ]),
    ("Reward, trained on all 11 assets", [
        ("multi_all", "Log return"),
        ("rw_vol_scaled", "Volatility-scaled return  (kept)"),
        ("rw_dsr", "Differential Sharpe ratio"),
        ("rw_drawdown", "Drawdown penalty"),
        ("rw_downside", "Losses counted double"),
    ]),
    ("DQN upgrades, on the kept setup", [
        ("rw_vol_scaled", "Plain Double DQN  (kept)"),
        ("dueling", "Dueling network"),
        ("nstep3", "3-step returns"),
        ("per", "Prioritised replay"),
        ("combo", "All three"),
    ]),
]


def ablations() -> None:
    all_assets = pd.read_csv(RESULTS / "BTC-USD/rw_vol_scaled/val_assets.csv")
    symbols = list(all_assets.symbol.unique())
    random_ref = (all_assets[KEY] - all_assets["edge_vs_random"]).groupby(all_assets.symbol).first()
    cfg = RunConfig(min_hold=10, window=3)
    sma_edge = sum(
        metrics(run_policy(make_env(cfg, "val", symbol=s), sma_crossover()))[KEY] - random_ref[s]
        for s in symbols) / len(symbols)

    rows = []
    for group, runs in ABLATIONS:
        for name, label in runs:
            m = pd.read_csv(RESULTS / f"BTC-USD/{name}/val_multi_metrics.csv")
            mid, lo, hi = seed_summary(m["edge_vs_random"])
            rows.append({"group": group, "run": name, "label": label, "edge_vs_random": mid,
                         "lo": lo, "hi": hi, "trades_per_asset": iqm(m["turnover"]),
                         "max_drawdown": iqm(m["max_drawdown"])})
    table = pd.DataFrame(rows)
    table.assign(sma_edge_vs_random=sma_edge).round(4).to_csv(TABLES / "ablations.csv", index=False)

    fig, ax = new_axes(10.5, 7.2)
    y, ticks, labels = 0.0, [], []
    for group, g in table.groupby("group", sort=False):
        ax.text(-0.012, y, group, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=10, color=INK, fontweight="bold")
        y -= 1
        for _, r in g.iterrows():
            kept = "(kept)" in r["label"]
            color = BLUE if kept else MUTED
            ax.hlines(y, r["lo"], r["hi"], color=color, lw=2, zorder=3)
            ax.scatter(r["edge_vs_random"], y, s=70, color=color, edgecolor=SURFACE,
                       linewidth=2, zorder=4)
            ticks.append(y)
            labels.append(r["label"])
            y -= 1
        y -= 0.5
    ax.axvline(0, color=AXIS, lw=1, zorder=2)
    ax.axvline(sma_edge, color=AQUA, lw=1.5, zorder=2)
    ax.text(sma_edge + 0.002, 0.55, f"SMA 20/50 rule {sma_edge:+.2f}", ha="left", va="bottom",
            fontsize=9, color=SECONDARY)
    ax.text(0.002, 0.55, "random agents", ha="left", va="bottom", fontsize=9, color=MUTED)
    ax.set_xlim(-0.005, table["hi"].max() + 0.01)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", labelcolor=SECONDARY)
    ax.set_ylim(y + 0.8, 1.3)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}"))
    titles(fig, ax, "No single change clearly improved the agent; the intervals all overlap",
           "Edge over random agents on the validation period (2022-2023), log return per year, "
           "averaged over 11 assets.\nDot: interquartile mean of 10 seeds; line: 95% interval. "
           "Blue: the setting carried forward.")
    save(fig, "ablations.png")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    walk_forward()
    selection_bias()
    ablations()
