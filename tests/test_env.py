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


def test_multi_asset_env_uses_every_asset():
    from rltrader.env import MultiAssetEnv

    envs = [TradingEnv(make_df(seed=s), episode_length=20) for s in range(3)]
    multi = MultiAssetEnv(envs)
    multi.reset(seed=0)
    seen = set()
    for _ in range(40):
        obs, _ = multi.reset()
        assert obs.shape == multi.observation_space.shape
        seen.add(envs.index(multi.current))
        _, _, terminated, truncated, info = multi.step(1)
        assert info["value"] == multi.current.value
    assert seen == {0, 1, 2}


def test_metrics_use_the_calendar_for_years():
    # a stock-like series: 5 bars a week for two years, 10% growth per year
    idx = pd.bdate_range("2020-01-01", "2021-12-31", name="date")
    years = (idx[-1] - idx[0]).days / 365.25
    value = 1.1 ** (np.arange(len(idx)) / (len(idx) - 1) * years)
    record = pd.DataFrame({"value": value, "position": 1.0, "traded": 0.0}, index=idx)
    assert metrics(record)["cagr"] == pytest.approx(0.10, abs=1e-6)


def _play(env, actions):
    env.reset(seed=0)
    out = [env.step(a) for a in actions]
    return [o[1] for o in out], [o[4]["value"] for o in out]


def test_reward_types_change_the_lesson_but_never_the_money():
    from rltrader.env import REWARD_TYPES, TARGET_VOL

    df, actions = make_df(), [1] * 40 + [0] * 5 + [1] * 40
    plain_r, plain_v = _play(TradingEnv(df), actions)
    for kind in REWARD_TYPES:
        rewards, values = _play(TradingEnv(df, reward_type=kind), actions)
        assert values == plain_v
        assert np.all(np.isfinite(rewards))

    # downside: gains unchanged, losses doubled
    down_r, _ = _play(TradingEnv(df, reward_type="downside", reward_param=1.0), actions)
    assert down_r == pytest.approx([r * 2 if r < 0 else r for r in plain_r])

    # vol_scaled: each step's return divided by that day's volatility
    env = TradingEnv(df, reward_type="vol_scaled")
    vol_r, _ = _play(env, actions)
    t0 = env.first_t
    expected = [r * TARGET_VOL / env.vol[t0 + i] for i, r in enumerate(plain_r)]
    assert vol_r == pytest.approx(expected)

    # drawdown: never more than the plain reward, and strictly less at some point
    dd_r, _ = _play(TradingEnv(df, reward_type="drawdown", reward_param=1.0), actions)
    assert all(d <= p + 1e-12 for d, p in zip(dd_r, plain_r)) and sum(dd_r) < sum(plain_r)

    # differential Sharpe: the very first step is simply return / volatility
    dsr_env = TradingEnv(df, reward_type="dsr")
    dsr_r, _ = _play(dsr_env, actions)
    assert dsr_r[0] == pytest.approx(plain_r[0] / dsr_env.vol[t0] * TARGET_VOL)


def test_walk_forward_folds_never_overlap_and_gbm_labels_stay_inside_training(tmp_path, monkeypatch):
    from rltrader.experiment import RunConfig
    from rltrader.walkforward import fold_envs, gbm_fold, make_folds

    (tmp_path / "data").mkdir()
    make_df(n=900, seed=1).to_csv(tmp_path / "data" / "A.csv")   # 2020-01-01 .. mid 2022
    make_df(n=900, seed=2).iloc[500:].to_csv(tmp_path / "data" / "B.csv")  # starts mid 2021
    monkeypatch.chdir(tmp_path)

    cfg = RunConfig(symbol="A", train_symbols=("A", "B"), eval_symbols=("A", "B"),
                    window=3, min_hold=10)
    fold = make_folds(2022, 2022)[0]
    train, test = fold_envs(cfg, fold, ("A", "B"), "train"), fold_envs(cfg, fold, ("A", "B"), "test")
    assert set(train) == {"A"}          # B has too little history before 2022 to learn from
    assert set(test) == {"A", "B"}      # but both can be traded in 2022
    assert train["A"].dates[train["A"].last_t] <= pd.Timestamp("2021-12-31")
    assert test["A"].dates[test["A"].first_t] >= pd.Timestamp("2022-01-01")

    rows = gbm_fold(cfg, 0, fold)
    assert {r["symbol"] for r in rows} == {"A", "B"}
    assert all(np.isfinite(r["log_return_per_year"]) for r in rows)
