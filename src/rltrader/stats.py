"""Summarising results across random seeds.

RL results vary a lot from seed to seed, so one number per method is not
enough. Following Agarwal et al. (2021, "Deep RL at the Edge of the
Statistical Precipice") we report:

  * the interquartile mean (IQM): drop the best 25% and worst 25% of seeds
    and average the rest. Less swayed by one lucky or unlucky seed than the
    mean, and uses more of the data than the median;
  * a bootstrap confidence interval: resample the seeds with replacement many
    times and see how much the IQM moves. A wide interval means "we do not
    really know yet";
  * the probability of improvement: how often a seed of method B beats a seed
    of method A.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import trim_mean


def iqm(x) -> float:
    return float(trim_mean(np.asarray(x, dtype=float), 0.25))


def bootstrap_ci(x, stat=iqm, n_boot: int = 10_000, level: float = 0.95, seed: int = 0):
    """(low, high) interval for ``stat`` of ``x``."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), (n_boot, len(x)))]
    stats = np.array([stat(s) for s in samples])
    tail = (1 - level) / 2 * 100
    return float(np.percentile(stats, tail)), float(np.percentile(stats, 100 - tail))


def compare(a, b, n_boot: int = 10_000, level: float = 0.95, seed: int = 0) -> dict:
    """How much better is ``b`` than ``a``?  Positive difference = b is higher."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.array([
        iqm(b[rng.integers(0, len(b), len(b))]) - iqm(a[rng.integers(0, len(a), len(a))])
        for _ in range(n_boot)
    ])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    # share of (a seed, b seed) pairs where b wins; ties count half
    wins = (b[:, None] > a[None, :]).mean() + 0.5 * (b[:, None] == a[None, :]).mean()
    return {
        "difference": iqm(b) - iqm(a),
        "ci_low": float(low),
        "ci_high": float(high),
        "prob_improvement": float(wins),
        # the interval excludes zero: the difference is larger than seed noise
        "clear": bool(low > 0 or high < 0),
    }
