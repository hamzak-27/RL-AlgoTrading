"""One training run for one seed. Kept in the package (not in a script) so
several seeds can run in parallel worker processes."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import Split, load
from .dqn import DQNAgent, DQNConfig
from .env import TradingEnv
from .evaluate import agent_policy, metrics, run_policy


@dataclass
class RunConfig:
    symbol: str = "BTC-USD"
    steps: int = 60_000
    eval_every: int = 2_000
    window: int = 10
    fee: float = 0.001
    episode_length: int = 365
    switch_penalty: float = 0.0  # reward penalty per unit of position changed
    min_hold: int = 0            # minimum days to hold a position after a trade
    dqn: dict = field(default_factory=dict)  # overrides for DQNConfig

    def to_dict(self) -> dict:
        return asdict(self)


def make_env(cfg: RunConfig, period: str, episode_length: int | None = None) -> TradingEnv:
    df = load(Path("data") / f"{cfg.symbol}.csv")
    start, end = Split().periods(df)[period]
    return TradingEnv(df, window=cfg.window, fee=cfg.fee,
                      episode_length=episode_length, start=start, end=end,
                      switch_penalty=cfg.switch_penalty, min_hold=cfg.min_hold)


SMOOTH_WINDOW = 5


def choose_checkpoints(scores: list[float], window: int = SMOOTH_WINDOW) -> dict[str, int]:
    """Which evaluation's checkpoint to keep, under three rules.

    best:     the single highest validation score. Flattering: with 30 noisy
              evaluations, the highest one is partly luck (selection bias).
    smoothed: the end of the best *stretch* of ``window`` evaluations in a
              row. One lucky evaluation cannot win; the agent has to be good
              for a while.
    last:     the final model, no choosing at all. Its validation score is
              an unbiased measure, so this is the one to compare variants on.
    """
    s = np.asarray(scores, dtype=float)
    window = min(window, len(s))
    trailing = np.convolve(s, np.ones(window) / window, mode="valid")  # mean of s[i-window+1..i]
    return {
        "best": int(s.argmax()),
        "smoothed": int(trailing.argmax()) + window - 1,
        "last": len(s) - 1,
    }


def train_seed(cfg: RunConfig, seed: int, out_dir: str) -> dict:
    """Train one agent; return its validation curve and val/test results
    for each checkpoint-selection rule."""
    torch.set_num_threads(1)  # one core per worker, so workers do not fight
    train_env = make_env(cfg, "train", cfg.episode_length)
    val_env, test_env = make_env(cfg, "val"), make_env(cfg, "test")

    dqn_cfg = DQNConfig(**{"eps_decay_steps": cfg.steps // 2, **cfg.dqn})
    agent = DQNAgent(train_env.observation_space.shape[0], train_env.action_space.n, dqn_cfg, seed)

    curve, snapshots = [], []
    obs, _ = train_env.reset(seed=seed)
    for step in range(1, cfg.steps + 1):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        agent.observe(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = train_env.reset()

        if step % cfg.eval_every == 0:
            m = metrics(run_policy(val_env, agent_policy(agent)))
            curve.append({"seed": seed, "step": step, "val_log_return": m["log_return"],
                          "val_turnover": m["turnover"]})
            snapshots.append(copy.deepcopy(agent.q.state_dict()))

    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    chosen = choose_checkpoints([c["val_log_return"] for c in curve])
    result = {"seed": seed, "curve": curve, "val": {}, "test": {}, "test_equity": {}}
    for rule, i in chosen.items():
        agent.q.load_state_dict(snapshots[i])
        agent.save(ckpt_dir / f"seed{seed}_{rule}.pt")
        test_record = run_policy(test_env, agent_policy(agent))
        extra = {"chosen_step": curve[i]["step"]}
        result["val"][rule] = {**extra, **metrics(run_policy(val_env, agent_policy(agent)))}
        result["test"][rule] = {**extra, **metrics(test_record)}
        result["test_equity"][rule] = test_record["value"]
    return result
