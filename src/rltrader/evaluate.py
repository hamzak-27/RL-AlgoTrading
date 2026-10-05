"""Running a policy through the environment, scoring it, and simple baselines.

Every strategy (the RL agent and each baseline) goes through the *same*
environment, so they all pay the same fees and trade at the same prices.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .env import TradingEnv

Policy = Callable[[np.ndarray, TradingEnv], int]


def run_policy(env: TradingEnv, policy: Policy, seed: int = 0) -> pd.DataFrame:
    """Play one full episode and return the day-by-day record."""
    obs, _ = env.reset(seed=seed)
    rows = [{"date": env.dates[env.t], "value": 1.0, "position": 0.0, "traded": 0.0}]
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs, env))
        rows.append(info)
        done = terminated or truncated
    return pd.DataFrame(rows).set_index("date")


def metrics(record: pd.DataFrame) -> dict[str, float]:
    """Standard performance numbers from a portfolio-value series."""
    value = record["value"]
    rets = value.pct_change().dropna()
    # Measured from the calendar, so it is right for crypto (365 bars a year)
    # and for stocks (about 252) alike.
    years = (record.index[-1] - record.index[0]).days / 365.25
    periods_per_year = len(rets) / years if years > 0 else 0.0
    total = value.iloc[-1] / value.iloc[0] - 1.0
    cagr = (value.iloc[-1] / value.iloc[0]) ** (1 / years) - 1.0 if years > 0 else 0.0
    std = rets.std()
    downside = rets[rets < 0].std()
    ann = np.sqrt(periods_per_year)
    max_dd = (value / value.cummax() - 1.0).min()
    return {
        "total_return": total,
        # log of final wealth: what the agent's rewards add up to
        "log_return": float(np.log(value.iloc[-1] / value.iloc[0])),
        # the same per year, so periods of different length can be compared
        "log_return_per_year": float(np.log(value.iloc[-1] / value.iloc[0]) / years) if years > 0 else 0.0,
        "cagr": cagr,
        # return per unit of risk (0 if the strategy never held anything)
        "sharpe": rets.mean() / std * ann if std > 0 else 0.0,
        # like Sharpe, but only downside moves count as risk
        "sortino": rets.mean() / downside * ann if downside > 0 else 0.0,
        # worst peak-to-trough loss
        "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else 0.0,
        # share of days holding the asset, and total position changed
        "exposure": float((record["position"] != 0).mean()),
        "turnover": float(record["traded"].sum()),
    }


# ------------------------------------------------------------------ baselines
def _action_for(env: TradingEnv, position: float) -> int:
    return int(np.argmin(np.abs(env.positions - position)))


def buy_and_hold(obs: np.ndarray, env: TradingEnv) -> int:
    return _action_for(env, 1.0)


def always_flat(obs: np.ndarray, env: TradingEnv) -> int:
    return _action_for(env, 0.0)


def sma_crossover(fast: int = 20, slow: int = 50) -> Policy:
    """Hold the asset while the fast moving average is above the slow one."""

    def policy(obs: np.ndarray, env: TradingEnv) -> int:
        closes = env.close[max(0, env.t - slow + 1) : env.t + 1]  # up to today only
        if len(closes) < slow:
            return _action_for(env, 0.0)
        return _action_for(env, 1.0 if closes[-fast:].mean() > closes.mean() else 0.0)

    return policy


def random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs, env: int(rng.integers(env.action_space.n))


def agent_policy(agent) -> Policy:
    return lambda obs, env: agent.act(obs, greedy=True)
