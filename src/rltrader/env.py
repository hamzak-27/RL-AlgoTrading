"""A single-asset trading environment with honest accounting.

One step = one daily bar. The timeline of a step is:

    close of day t        agent sees features up to day t, picks a target position
    open  of day t+1      the trade is executed here (it cannot trade at a
                          price it has already seen), and the fee is charged
    close of day t+1      portfolio is valued; reward = log growth of the
                          portfolio over the whole step

Fixes from the 2023 notebook:
  * the agent's own position is part of the observation;
  * actions are *target positions*, so "buy forever" is impossible;
  * losses count (reward is the log return of the portfolio, not max(gain, 0));
  * fees are charged on every change of position;
  * running out of data is a time limit (``truncated``), not a real ending
    (``terminated``), which matters for how the agent learns values.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .features import VOL_WINDOW, build_features

REWARD_TYPES = ("log_return", "vol_scaled", "dsr", "drawdown", "downside")
TARGET_VOL = 0.03   # a typical daily move for crypto; keeps all rewards on one scale
DSR_ETA = 0.01      # how quickly the differential Sharpe "forgets" old returns


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 10,
        fee: float = 0.001,
        positions: tuple[float, ...] = (0.0, 1.0),
        episode_length: int | None = None,
        start: str | None = None,
        end: str | None = None,
        switch_penalty: float = 0.0,
        min_hold: int = 0,
        reward_type: str = "log_return",
        reward_param: float = 1.0,
    ):
        """
        df:             bars with open/high/low/close/volume, ascending by date
        window:         how many past days of features the agent sees
        fee:            cost per unit of position changed (0.001 = 0.1%)
        positions:      what each action means, as a fraction of the portfolio
                        held in the asset: (0, 1) = flat/long,
                        (-1, 0, 1) = short/flat/long
        episode_length: if set, each reset starts at a random date and runs for
                        this many days (training). If None, an episode walks
                        the whole period once from the start (evaluation).
        switch_penalty: extra amount subtracted from the *reward* (not from the
                        portfolio) per unit of position changed. It teaches
                        the agent that switching must be worth it; the money
                        it actually has is still computed with ``fee`` only.
        min_hold:       after a trade the position is locked until it has been
                        held this many days. When > 0, the observation gets one
                        extra number: the share of the lock still remaining.
        reward_type:    what the agent is taught to maximise. The money is
                        always computed the same way; only the lesson differs.
                        "log_return"  growth of the portfolio (the default)
                        "vol_scaled"  growth divided by how volatile the asset
                                      is right now, so a calm asset and a wild
                                      one teach equally loud lessons
                        "dsr"         differential Sharpe ratio (Moody & Saffell,
                                      1998): did this step raise or lower the
                                      running return-per-unit-of-risk?
                        "drawdown"    growth, minus ``reward_param`` times any
                                      new fall below the portfolio's own peak
                        "downside"    growth, with losses counted
                                      (1 + ``reward_param``) times
        start, end:     restrict trading to this date range (inclusive). Pass
                        the full price history as ``df`` and choose the period
                        here, so features at the start of a period can still
                        look back at earlier bars.
        """
        super().__init__()
        feats = build_features(df)
        self.dates = feats.index
        self.features = feats.to_numpy(dtype=np.float32)
        self.open = df.loc[self.dates, "open"].to_numpy()
        self.close = df.loc[self.dates, "close"].to_numpy()
        logret = np.log(df["close"]).diff()
        self.vol = logret.rolling(VOL_WINDOW).std().loc[self.dates].to_numpy()  # backward-looking
        if reward_type not in REWARD_TYPES:
            raise ValueError(f"reward_type must be one of {REWARD_TYPES}")
        self.reward_type, self.reward_param = reward_type, reward_param

        self.window = window
        self.fee = fee
        self.positions = np.asarray(positions, dtype=np.float64)
        self.episode_length = episode_length
        self.switch_penalty = switch_penalty
        self.min_hold = min_hold

        self.first_t = window - 1          # earliest day with a full window
        self.last_t = len(self.dates) - 1  # last day we have prices for
        if start is not None:
            self.first_t = max(self.first_t, int(self.dates.searchsorted(pd.Timestamp(start))))
        if end is not None:
            self.last_t = int(self.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
        if self.last_t - self.first_t < 2:
            raise ValueError("Not enough data for this window size")

        # +1 for current position, +1 for the holding lock if it is in use
        obs_dim = window * self.features.shape[1] + 1 + (1 if min_hold > 0 else 0)
        self.action_space = spaces.Discrete(len(self.positions))
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        block = self.features[self.t - self.window + 1 : self.t + 1].ravel()
        extra = [self.position]
        if self.min_hold > 0:
            extra.append(self.lock / self.min_hold)
        return np.concatenate([block, np.asarray(extra, dtype=np.float32)])

    def _shape(self, r: float, t: int) -> float:
        """Turn this step's log return ``r`` into the reward the agent learns from."""
        kind = self.reward_type
        if kind == "vol_scaled":
            return r * TARGET_VOL / self.vol[t]
        if kind == "downside":
            return r * (1.0 + self.reward_param) if r < 0 else r
        if kind == "drawdown":
            self.peak = max(self.peak, self.value)
            new_drawdown = 1.0 - self.value / self.peak
            deeper = max(0.0, new_drawdown - self.drawdown)
            self.drawdown = new_drawdown
            return r - self.reward_param * deeper
        if kind == "dsr":
            a, b = self.dsr_a, self.dsr_b
            da, db = r - a, r * r - b
            d = (b * da - 0.5 * a * db) / max(b - a * a, 1e-10) ** 1.5
            self.dsr_a, self.dsr_b = a + DSR_ETA * da, b + DSR_ETA * db
            return float(np.clip(d, -10.0, 10.0)) * TARGET_VOL
        return r

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.episode_length is None:
            self.t = self.first_t
            self.end_t = self.last_t
        else:
            latest_start = max(self.first_t, self.last_t - self.episode_length)
            self.t = int(self.np_random.integers(self.first_t, latest_start + 1))
            self.end_t = min(self.t + self.episode_length, self.last_t)
        self.position = 0.0
        self.lock = 0  # days the current position must still be held
        self.value = 1.0  # portfolio value, starting from 1 unit of cash
        self.peak, self.drawdown = 1.0, 0.0             # for the "drawdown" reward
        self.dsr_a, self.dsr_b = 0.0, self.vol[self.t] ** 2  # running mean of r and of r^2
        return self._obs(), {}

    def step(self, action: int):
        target = float(self.positions[int(action)])
        if self.lock > 0:  # still inside the minimum holding period
            target = self.position
            self.lock -= 1
        t = self.t
        value_before = self.value

        # 1) overnight: the old position is carried from close[t] to open[t+1]
        self.value *= 1.0 + self.position * (self.open[t + 1] / self.close[t] - 1.0)
        # 2) trade at the open and pay the fee on the amount changed
        traded = abs(target - self.position)
        self.value *= 1.0 - self.fee * traded
        # 3) intraday: the new position is carried from open[t+1] to close[t+1]
        self.value *= 1.0 + target * (self.close[t + 1] / self.open[t + 1] - 1.0)

        if traded > 0 and self.min_hold > 0:
            self.lock = self.min_hold - 1
        self.position = target
        self.t = t + 1

        terminated = self.value <= 0.0  # wiped out: a true ending
        truncated = (self.t >= self.end_t) and not terminated  # ran out of time
        reward = -10.0 if terminated else self._shape(float(np.log(self.value / value_before)), t)
        reward -= self.switch_penalty * traded

        info = {
            "date": self.dates[self.t],
            "value": self.value,
            "position": self.position,
            "traded": traded,
        }
        return self._obs(), reward, terminated, truncated, info


class MultiAssetEnv(gym.Env):
    """Training on several assets at once: every episode is played on one
    asset picked at random.

    The agent never learns which asset it is on. It only sees the features,
    which are scaled the same way for every asset, so whatever it learns has
    to work across all of them. That is the point: a pattern that only exists
    in one coin's history is more likely an accident than a pattern that
    shows up in eight.
    """

    metadata = {"render_modes": []}

    def __init__(self, envs: list[TradingEnv]):
        super().__init__()
        self.envs = envs
        self.current = envs[0]
        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current = self.envs[int(self.np_random.integers(len(self.envs)))]
        return self.current.reset(seed=int(self.np_random.integers(2**31 - 1)))

    def step(self, action: int):
        return self.current.step(action)
