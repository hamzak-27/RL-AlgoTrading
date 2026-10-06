import numpy as np
import pytest
import torch

from rltrader.dqn import DQNAgent, DQNConfig, QNetwork, ReplayBuffer


def test_n_step_sums_rewards_and_respects_episode_ends():
    cfg = DQNConfig(n_step=3, gamma=0.5, reward_scale=1.0, learning_starts=10_000)
    agent = DQNAgent(obs_dim=1, n_actions=2, cfg=cfg)
    obs = lambda i: np.array([i], np.float32)
    rewards = [1.0, 2.0, 4.0, 8.0, 16.0]
    for i, r in enumerate(rewards):
        agent.observe(obs(i), 0, r, obs(i + 1), terminated=False, truncated=(i == 4))
    buf = agent.buffer
    assert buf.size == 5  # every step ends up stored exactly once
    # full 3-step returns: r0 + 0.5 r1 + 0.25 r2, landing 3 states later
    assert buf.rewards[0] == pytest.approx(1 + 0.5 * 2 + 0.25 * 4)
    assert buf.next_obs[0, 0] == 3 and buf.discounts[0] == pytest.approx(0.5 ** 3)
    assert buf.rewards[2] == pytest.approx(4 + 0.5 * 8 + 0.25 * 16)
    # the episode ended, so the last two are cut short and stop at the final state
    assert buf.rewards[3] == pytest.approx(8 + 0.5 * 16) and buf.discounts[3] == pytest.approx(0.25)
    assert buf.rewards[4] == pytest.approx(16) and buf.discounts[4] == pytest.approx(0.5)
    assert all(buf.next_obs[i, 0] == 5 for i in (2, 3, 4))
    assert not agent.pending  # nothing leaks into the next episode


def test_one_step_is_the_plain_case():
    agent = DQNAgent(1, 2, DQNConfig(reward_scale=1.0, learning_starts=10_000))
    agent.observe(np.zeros(1, np.float32), 1, 0.3, np.ones(1, np.float32), False)
    assert agent.buffer.rewards[0] == pytest.approx(0.3)
    assert agent.buffer.discounts[0] == pytest.approx(agent.cfg.gamma)


def test_prioritised_buffer_favours_surprising_transitions():
    buf = ReplayBuffer(100, 1, np.random.default_rng(0), alpha=1.0)
    for i in range(100):
        buf.add(np.zeros(1), 0, 0.0, np.zeros(1), 0.0, 0.99)
    errors = np.full(100, 0.01)
    errors[7] = 10.0  # one transition the network got very wrong
    buf.update_priorities(np.arange(100), errors)
    _, weights, idx = buf.sample(2000, beta=1.0)
    assert (idx == 7).mean() > 0.5                      # sampled far more often...
    assert weights[idx == 7].max() < weights[idx != 7].min()  # ...but weighted down


def test_dueling_network_shape_and_learning_step():
    net = QNetwork(obs_dim=5, n_actions=3, hidden=8, dueling=True)
    assert net(torch.zeros(4, 5)).shape == (4, 3)
    cfg = DQNConfig(dueling=True, per=True, n_step=3, learning_starts=70, batch_size=16)
    agent = DQNAgent(5, 3, cfg)
    rng = np.random.default_rng(0)
    loss = None
    for i in range(120):
        loss = agent.observe(rng.normal(size=5).astype(np.float32), int(rng.integers(3)),
                             float(rng.normal()) * 0.01, rng.normal(size=5).astype(np.float32),
                             False, truncated=(i % 30 == 29))
    assert loss is not None and np.isfinite(loss)
