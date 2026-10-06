"""Walk-forward testing: retrain, test on the next year, move forward, repeat.

One fixed train/test split gives one answer, and that answer depends on what
the market happened to do in that one test period. Walk-forward repeats the
whole exercise for every year:

    train on everything up to 2018  ->  test on 2019
    train on everything up to 2019  ->  test on 2020
    ...

Each test year is unseen by the model that trades it, exactly as it would be
in real use, and we get one result per year instead of one in total.

Two methods run through the same folds, the same environment and the same
fees, so they can be compared directly:
  * "dqn": the reinforcement-learning agent;
  * "gbm": a supervised baseline. A gradient-boosted tree model (LightGBM)
    predicts the return over the next holding period from the same features,
    and the rule is simply "hold the asset when the prediction is positive".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dqn import DQNAgent, DQNConfig
from .env import MultiAssetEnv, TradingEnv
from .evaluate import agent_policy, metrics, run_policy
from .experiment import RunConfig, make_env

MIN_TRAIN_DAYS = 250  # an asset joins a fold only with about a year of history to learn from
MIN_TEST_DAYS = 100


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: str
    test_start: str
    test_end: str


def make_folds(first_year: int, last_year: int) -> list[Fold]:
    return [Fold(str(y), f"{y - 1}-12-31", f"{y}-01-01", f"{y}-12-31")
            for y in range(first_year, last_year + 1)]


def _days(env: TradingEnv) -> int:
    return env.last_t - env.first_t


def fold_envs(cfg: RunConfig, fold: Fold, symbols, period: str,
              episode_length: int | None = None) -> dict[str, TradingEnv]:
    """The environments of one fold, for the assets that have enough data in it."""
    start, end = (None, fold.train_end) if period == "train" else (fold.test_start, fold.test_end)
    minimum = MIN_TRAIN_DAYS if period == "train" else MIN_TEST_DAYS
    envs = {}
    for s in symbols:
        try:
            env = make_env(cfg, None, episode_length, s, start=start, end=end)
        except ValueError:  # the asset did not exist yet
            continue
        if _days(env) >= minimum:
            envs[s] = env
    return envs


def _score(fold: Fold, seed: int, test_envs: dict, policy_for) -> list[dict]:
    return [{"fold": fold.name, "seed": seed, "symbol": s,
             **metrics(run_policy(env, policy_for(s, env)))}
            for s, env in test_envs.items()]


# ------------------------------------------------------------------------ DQN
def dqn_fold(cfg: RunConfig, seed: int, fold: Fold) -> list[dict]:
    torch.set_num_threads(1)
    symbols = cfg.train_symbols or (cfg.symbol,)
    train_env = MultiAssetEnv(list(
        fold_envs(cfg, fold, symbols, "train", cfg.episode_length).values()))
    dqn_cfg = DQNConfig(**{"eps_decay_steps": cfg.steps // 2, "per_beta_steps": cfg.steps,
                           **cfg.dqn})
    agent = DQNAgent(train_env.observation_space.shape[0], train_env.action_space.n, dqn_cfg, seed)

    obs, _ = train_env.reset(seed=seed)
    for _ in range(cfg.steps):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = train_env.reset()

    test_envs = fold_envs(cfg, fold, cfg.eval_symbols or symbols, "test")
    return _score(fold, seed, test_envs, lambda s, env: agent_policy(agent))


# ------------------------------------------------------- supervised baseline
def _window_features(env: TradingEnv, t: np.ndarray) -> np.ndarray:
    """The same market features the agent sees at each day in ``t`` (no position)."""
    return np.stack([env.features[i - env.window + 1: i + 1].ravel() for i in t])


def gbm_fold(cfg: RunConfig, seed: int, fold: Fold) -> list[dict]:
    from lightgbm import LGBMRegressor

    horizon = max(cfg.min_hold, 1)  # predict over the period a trade must be held
    symbols = cfg.train_symbols or (cfg.symbol,)
    xs, ys = [], []
    for env in fold_envs(cfg, fold, symbols, "train").values():
        # Stop ``horizon`` days before the end of training: a label there would
        # need prices from inside the test year (this gap is called purging).
        t = np.arange(env.first_t, env.last_t - horizon + 1)
        # What buying at tomorrow's open and holding for the horizon returned,
        # in units of the asset's own volatility.
        label = np.log(env.close[t + horizon] / env.open[t + 1]) / (env.vol[t] * np.sqrt(horizon))
        xs.append(_window_features(env, t))
        ys.append(label)
    model = LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=15,
                          min_child_samples=100, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
    model.fit(np.concatenate(xs), np.concatenate(ys))

    def policy_for(symbol: str, env: TradingEnv):
        t = np.arange(env.first_t, env.last_t + 1)
        go_long = dict(zip(t, model.predict(_window_features(env, t)) > 0))
        target = {flag: int(np.argmin(np.abs(env.positions - (1.0 if flag else 0.0))))
                  for flag in (True, False)}
        return lambda obs, env: target[bool(go_long[env.t])]

    test_envs = fold_envs(cfg, fold, cfg.eval_symbols or symbols, "test")
    return _score(fold, seed, test_envs, policy_for)


METHODS = {"dqn": dqn_fold, "gbm": gbm_fold}
