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

Optional upgrades (the main pieces of "Rainbow", Hessel et al. 2018), all off
by default so each can be measured on its own:
  * dueling network   (``dueling``)
  * n-step returns    (``n_step``)
  * prioritised replay (``per``)
"""
from __future__ import annotations

from collections import deque
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
    # --- regularisation: ways to stop the network memorising the training years
    weight_decay: float = 0.0        # pulls weights towards zero (simpler functions)
    layer_norm: bool = False         # normalises each hidden layer's activations
    obs_noise: float = 0.0           # random noise added to states when learning
    # --- DQN upgrades
    dueling: bool = False            # separate "how good is this state" from "which action"
    n_step: int = 1                  # look this many real rewards ahead before guessing
    per: bool = False                # replay surprising transitions more often
    per_alpha: float = 0.6           # 0 = uniform sampling, 1 = fully by surprise
    per_beta_start: float = 0.4      # how much to correct for the uneven sampling...
    per_beta_steps: int = 60_000     # ...rising to full correction over this many steps


class ReplayBuffer:
    """Stores past transitions and hands back random mini-batches.

    With ``alpha > 0`` it is a prioritised buffer (Schaul et al., 2016): a
    transition is sampled in proportion to how wrong the network last was
    about it, so learning time goes where there is most to learn.
    """

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator, alpha: float = 0.0):
        self.capacity, self.rng, self.alpha = capacity, rng, alpha
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.terminated = np.zeros(capacity, np.float32)
        self.discounts = np.zeros(capacity, np.float32)   # gamma^(steps looked ahead)
        self.priorities = np.zeros(capacity, np.float64)
        self.size = self.pos = 0

    def add(self, obs, action, reward, next_obs, terminated, discount):
        i = self.pos
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i], self.terminated[i] = action, reward, terminated
        self.discounts[i] = discount
        # New transitions get the highest priority so each is replayed at least once.
        self.priorities[i] = self.priorities[: self.size].max() if self.size else 1.0
        self.pos = (i + 1) % self.capacity  # overwrite the oldest when full
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 1.0):
        """Returns (batch tensors, importance weights, indices)."""
        if self.alpha > 0:
            cumulative = np.cumsum(self.priorities[: self.size])
            idx = np.searchsorted(cumulative, self.rng.random(batch_size) * cumulative[-1])
            idx = np.minimum(idx, self.size - 1)
            prob = self.priorities[idx] / cumulative[-1]
            # Over-sampled transitions get a smaller weight in the loss, so the
            # network's answers are not skewed towards the surprising cases.
            weights = (self.size * prob) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)
        else:
            idx = self.rng.integers(0, self.size, batch_size)
            weights = np.ones(batch_size, np.float32)
        batch = tuple(
            torch.as_tensor(x[idx])
            for x in (self.obs, self.actions, self.rewards, self.next_obs,
                      self.terminated, self.discounts)
        )
        return batch, torch.as_tensor(weights), idx

    def update_priorities(self, idx: np.ndarray, td_errors: np.ndarray) -> None:
        if self.alpha > 0:
            self.priorities[idx] = (np.abs(td_errors) + 1e-3) ** self.alpha


class QNetwork(nn.Module):
    """With ``dueling`` (Wang et al., 2016) the network outputs one number for
    "how good is it to be here" (V) and one per action for "how much better
    than average is this action" (A), and Q = V + A - mean(A). In trading most
    days the action barely matters, so learning V separately is efficient."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int, layer_norm: bool = False,
                 dueling: bool = False):
        super().__init__()
        norm = (lambda: nn.LayerNorm(hidden)) if layer_norm else nn.Identity
        self.dueling = dueling
        if dueling:
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden), norm(), nn.ReLU(),
                nn.Linear(hidden, hidden), norm(), nn.ReLU(),
            )
            self.value = nn.Linear(hidden, 1)
            self.advantage = nn.Linear(hidden, n_actions)
        else:
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), norm(), nn.ReLU(),
                nn.Linear(hidden, hidden), norm(), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.dueling:
            return self.net(x)
        h = self.body(x)
        advantage = self.advantage(h)
        return self.value(h) + advantage - advantage.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig | None = None, seed: int = 0):
        self.cfg = cfg or DQNConfig()
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        make_net = lambda: QNetwork(obs_dim, n_actions, self.cfg.hidden, self.cfg.layer_norm,
                                    self.cfg.dueling)
        self.q, self.q_target = make_net(), make_net()
        self.q_target.load_state_dict(self.q.state_dict())
        if self.cfg.weight_decay > 0:
            self.optim = torch.optim.AdamW(self.q.parameters(), lr=self.cfg.lr,
                                           weight_decay=self.cfg.weight_decay)
        else:
            self.optim = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.buffer = ReplayBuffer(self.cfg.buffer_size, obs_dim, self.rng,
                                   alpha=self.cfg.per_alpha if self.cfg.per else 0.0)
        self.pending: deque = deque()  # recent steps waiting to become n-step transitions
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
    def observe(self, obs, action, reward, next_obs, terminated, truncated=False) -> float | None:
        """Store one step and (once warmed up) do one learning step."""
        self._store(obs, action, reward * self.cfg.reward_scale, next_obs, terminated, truncated)
        self.steps += 1
        if self.steps < self.cfg.learning_starts:
            return None
        return self._learn()

    def _store(self, obs, action, reward, next_obs, terminated, truncated) -> None:
        """n-step returns (Sutton, 1988): instead of "one real reward, then the
        network's guess", store "n real rewards, then the guess". The guess is
        the unreliable part, so leaning on it less speeds up learning."""
        self.pending.append((obs, action, reward))
        episode_over = terminated or truncated
        while self.pending and (len(self.pending) == self.cfg.n_step or episode_over):
            total, discount = 0.0, 1.0
            for _, _, r in self.pending:
                total += discount * r
                discount *= self.cfg.gamma
            first_obs, first_action, _ = self.pending.popleft()
            # ``discount`` is now gamma^(number of rewards summed): how much the
            # guess about ``next_obs`` counts in the target.
            self.buffer.add(first_obs, first_action, total, next_obs, terminated, discount)

    def _learn(self) -> float:
        cfg = self.cfg
        beta = min(1.0, cfg.per_beta_start
                   + (1.0 - cfg.per_beta_start) * self.steps / cfg.per_beta_steps)
        batch, weights, idx = self.buffer.sample(cfg.batch_size, beta)
        obs, actions, rewards, next_obs, terminated, discounts = batch
        if cfg.obs_noise > 0:
            # The exact feature values of a past day will never repeat, so the
            # network should not rely on them to the last decimal.
            obs = obs + cfg.obs_noise * torch.randn_like(obs)
            next_obs = next_obs + cfg.obs_noise * torch.randn_like(next_obs)

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
            target = rewards + discounts * (1.0 - terminated) * next_value

        per_sample = nn.functional.smooth_l1_loss(q_pred, target, reduction="none")  # Huber
        loss = (weights * per_sample).mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), cfg.grad_clip)
        self.optim.step()
        self.buffer.update_priorities(idx, (target - q_pred).detach().numpy())

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
