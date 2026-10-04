"""Checks that the environment's accounting is right. If the simulator is
wrong, every result built on it is wrong, so this is the most important test file."""
import numpy as np
import pandas as pd
import pytest

from rltrader.data import load
from rltrader.env import TradingEnv
from rltrader.evaluate import always_flat, buy_and_hold, metrics, run_policy
from rltrader.features import build_features


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = rng.uniform(1e6, 2e6, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def test_load_sorts_newest_first_file(tmp_path):
    df = make_df(50).iloc[::-1]  # newest first, like the old bitcoindata.csv
    path = tmp_path / "x.csv"
    df.to_csv(path)
    assert load(path).index.is_monotonic_increasing


def test_features_do_not_look_ahead():
    df = make_df()
    full = build_features(df)
    cut = build_features(df.iloc[:300])  # same history, future removed
    pd.testing.assert_frame_equal(full.loc[cut.index], cut)


def test_staying_flat_keeps_value_at_one():
    record = run_policy(TradingEnv(make_df(), fee=0.001), always_flat)
    assert record["value"].eq(1.0).all()


def test_buy_and_hold_matches_hand_calculation():
    fee = 0.001
    env = TradingEnv(make_df(), fee=fee)
    record = run_policy(env, buy_and_hold)
    # buys at the first open after the start, pays the fee once, holds to the last close
    expected = (1 - fee) * env.close[env.last_t] / env.open[env.first_t + 1]
    assert record["value"].iloc[-1] == pytest.approx(expected)
    assert metrics(record)["turnover"] == pytest.approx(1.0)


def test_fees_are_charged_on_every_switch():
    df = make_df()
    flip = lambda obs, env: 1 - int(env.position)  # change position every day
    free = run_policy(TradingEnv(df, fee=0.0), flip)["value"].iloc[-1]
    costly = run_policy(TradingEnv(df, fee=0.01), flip)
    n_trades = int(costly["traded"].sum())
    assert costly["value"].iloc[-1] == pytest.approx(free * (1 - 0.01) ** n_trades)


def test_rewards_add_up_to_log_of_final_value():
    env = TradingEnv(make_df())
    env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        _, r, terminated, truncated, info = env.step(env.action_space.sample())
        total += r
        done = terminated or truncated
    assert truncated and not terminated  # running out of data is a time limit
    assert total == pytest.approx(np.log(info["value"]))


def test_observation_contains_position_and_date_range_is_respected():
    env = TradingEnv(make_df(), start="2020-09-01", end="2020-12-31", window=5)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape and obs[-1] == 0.0
    assert env.dates[env.t] >= pd.Timestamp("2020-09-01")
    obs, *_ = env.step(1)
    assert obs[-1] == 1.0
    record = run_policy(env, buy_and_hold)
    assert record.index[-1] <= pd.Timestamp("2020-12-31")


def test_random_training_episodes_stay_in_range():
    env = TradingEnv(make_df(), episode_length=30, end="2020-10-31")
    for seed in range(20):
        env.reset(seed=seed)
        done = False
        while not done:
            *_, terminated, truncated, info = env.step(1)
            done = terminated or truncated
        assert info["date"] <= pd.Timestamp("2020-10-31")


def test_switch_penalty_changes_reward_but_not_money():
    df = make_df()
    plain, penalised = TradingEnv(df), TradingEnv(df, switch_penalty=0.01)
    plain.reset(seed=0), penalised.reset(seed=0)
    for action in (1, 1, 0):  # buy, hold, sell
        _, r_plain, *_, info_plain = plain.step(action)
        _, r_pen, *_, info_pen = penalised.step(action)
        assert info_pen["value"] == info_plain["value"]
        assert r_pen == pytest.approx(r_plain - 0.01 * info_plain["traded"])


def test_min_hold_locks_the_position():
    env = TradingEnv(make_df(), min_hold=3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    positions = []
    for action in (1, 0, 0, 0, 0):  # buy, then try to sell every day
        obs, *_, info = env.step(action)
        positions.append(info["position"])
    # held for 3 days (the buy day plus two locked days), then the sell goes through
    assert positions == [1.0, 1.0, 1.0, 0.0, 0.0]
