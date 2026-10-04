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

from .features import build_features


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

        self.window = window
        self.fee = fee
        self.positions = np.asarray(positions, dtype=np.float64)
        self.episode_length = episode_length

        self.first_t = window - 1          # earliest day with a full window
        self.last_t = len(self.dates) - 1  # last day we have prices for
        if start is not None:
            self.first_t = max(self.first_t, int(self.dates.searchsorted(pd.Timestamp(start))))
        if end is not None:
            self.last_t = int(self.dates.searchsorted(pd.Timestamp(end), side="right")) - 1
        if self.last_t - self.first_t < 2:
            raise ValueError("Not enough data for this window size")

        obs_dim = window * self.features.shape[1] + 1  # +1 for current position
        self.action_space = spaces.Discrete(len(self.positions))
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        block = self.features[self.t - self.window + 1 : self.t + 1].ravel()
        return np.append(block, np.float32(self.position)).astype(np.float32)

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
        self.value = 1.0  # portfolio value, starting from 1 unit of cash
        return self._obs(), {}

    def step(self, action: int):
        target = float(self.positions[int(action)])
        t = self.t
        value_before = self.value

        # 1) overnight: the old position is carried from close[t] to open[t+1]
        self.value *= 1.0 + self.position * (self.open[t + 1] / self.close[t] - 1.0)
        # 2) trade at the open and pay the fee on the amount changed
        traded = abs(target - self.position)
        self.value *= 1.0 - self.fee * traded
        # 3) intraday: the new position is carried from open[t+1] to close[t+1]
        self.value *= 1.0 + target * (self.close[t + 1] / self.open[t + 1] - 1.0)

        self.position = target
        self.t = t + 1

        terminated = self.value <= 0.0  # wiped out: a true ending
        truncated = (self.t >= self.end_t) and not terminated  # ran out of time
        reward = -10.0 if terminated else float(np.log(self.value / value_before))

        info = {
            "date": self.dates[self.t],
            "value": self.value,
            "position": self.position,
            "traded": traded,
        }
        return self._obs(), reward, terminated, truncated, info
