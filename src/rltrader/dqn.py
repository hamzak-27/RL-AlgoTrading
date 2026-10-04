"""Deep Q-Network (DQN) agent in PyTorch.

Q(s, a) = "how much total future reward do I expect if I take action a in
state s and act well afterwards?"  The network learns Q; the policy is simply
"pick the action with the highest Q".

Fixes from the 2023 notebook:
  * replay buffer sampled *at random* (it used to train on the last 31 steps);
  * a separate, slowly-updated target network;
  * Double DQN targets (less over-optimistic value estimates);
  * one batched gradient step per environment step (it used to call
    ``fit`` 31 times per step, one sample at a time);
  * Huber loss + gradient clipping, so one wild reward cannot blow up training;
  * epsilon decays on a schedule over the whole run, not within 3 episodes;
  * time-limit endings still bootstrap from the next state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class DQNConfig:
    gamma: float = 0.99              # how much future reward counts vs. immediate
    lr: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    learning_starts: int = 1_000     # collect this many random steps before learning
    target_update_tau: float = 0.005  # how fast the target network follows
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000    # steps to go from eps_start to eps_end
    hidden: int = 64
    grad_clip: float = 10.0
    reward_scale: float = 100.0      # daily log-returns are ~0.01; bring them to ~1


class ReplayBuffer:
    """Stores past transitions and hands back random mini-batches."""

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.size = self.pos = 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i], self.terminated[i] = action, reward, terminated
        self.pos = (i + 1) % self.capacity  # overwrite the oldest when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, batch_size)
        return tuple(
            torch.as_tensor(x[idx])
            for x in (self.obs, self.actions, self.rewards, self.next_obs, self.terminated)
        )


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig | None = None, seed: int = 0):
        self.cfg = cfg or DQNConfig()
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.q = QNetwork(obs_dim, n_actions, self.cfg.hidden)
        self.q_target = QNetwork(obs_dim, n_actions, self.cfg.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, self.rng)
        self.steps = 0

    # ------------------------------------------------------------------ acting
    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.steps / self.cfg.eps_decay_steps)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy: mostly the best-known action, sometimes a random one."""
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            return int(self.q(torch.as_tensor(obs).unsqueeze(0)).argmax(dim=1).item())

    # ---------------------------------------------------------------- learning
    def observe(self, obs, action, reward, next_obs, terminated) -> float | None:
        """Store one transition and (once warmed up) do one learning step."""
        self.buffer.add(obs, action, reward * self.cfg.reward_scale, next_obs, terminated)
        self.steps += 1
        if self.steps < self.cfg.learning_starts:
            return None
        return self._learn()

    def _learn(self) -> float:
        cfg = self.cfg
        obs, actions, rewards, next_obs, terminated = self.buffer.sample(cfg.batch_size)

        # What the network currently predicts for the actions actually taken.
        q_pred = self.q(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        # What it *should* predict: reward now + discounted value of what follows.
        with torch.no_grad():
            # Double DQN: the online network chooses the next action,
            # the target network says how good that action is.
            best_next = self.q(next_obs).argmax(dim=1, keepdim=True)
            next_value = self.q_target(next_obs).gather(1, best_next).squeeze(1)
            # Only a real ending (terminated) cuts off the future. A time
            # limit does not, so those transitions are stored with terminated=0.
            target = rewards + cfg.gamma * (1.0 - terminated) * next_value

        loss = nn.functional.smooth_l1_loss(q_pred, target)  # Huber loss
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.optim.step()

        # Nudge the target network a little towards the online network.
        with torch.no_grad():
            for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
                p_targ.lerp_(p, cfg.target_update_tau)
        return float(loss.item())

    # ------------------------------------------------------------- persistence
    def save(self, path) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path) -> None:
        state = torch.load(path, map_location="cpu")
        self.q.load_state_dict(state)
        self.q_target.load_state_dict(state)
