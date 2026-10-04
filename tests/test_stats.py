import numpy as np

from rltrader.stats import bootstrap_ci, compare, iqm


def test_iqm_ignores_extreme_seeds():
    assert iqm([1, 2, 3, 4, 5, 6, 7, 1000]) == np.mean([3, 4, 5, 6])


def test_interval_contains_estimate_and_narrows_with_more_seeds():
    rng = np.random.default_rng(0)
    few, many = rng.normal(0, 1, 5), rng.normal(0, 1, 200)
    lo, hi = bootstrap_ci(few)
    assert lo <= iqm(few) <= hi
    lo2, hi2 = bootstrap_ci(many)
    assert hi2 - lo2 < hi - lo


def test_compare_separates_real_difference_from_noise():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 10)
    assert compare(a, rng.normal(5.0, 1.0, 10))["clear"]          # big real gap
    assert not compare(a, rng.normal(0.1, 1.0, 10))["clear"]      # lost in noise
    assert compare(a, a)["prob_improvement"] == 0.5


def test_checkpoint_rules():
    from rltrader.experiment import choose_checkpoints

    # index 2 is one lucky spike; indices 5-9 are a sustained good stretch
    scores = [-0.5, -0.4, 0.9, -0.6, -0.5, 0.2, 0.3, 0.25, 0.3, 0.2, -0.1]
    chosen = choose_checkpoints(scores, window=5)
    assert chosen["best"] == 2          # falls for the spike
    assert chosen["smoothed"] == 9      # end of the sustained stretch
    assert chosen["last"] == len(scores) - 1
