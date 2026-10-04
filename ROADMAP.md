# Improvement roadmap

Starting point (2026-10-04, BTC-USD, test period 2024-01-01 to 2026-10-03, 0.1% fee):

| Strategy | Total return | Sharpe | Trades |
|---|---|---|---|
| Buy & hold | +91.6% | 0.73 | 1 |
| SMA 20/50 | +55.6% | 0.64 | 27 |
| DQN (mean of 5 seeds) | +21.5% | 0.31 | 369 |

Known problems: results swing from -29% to +70% across seeds, the agent trades far
too often, and validation scores get worse the longer it trains (overfitting).

Rule for every step: change one thing, rerun all seeds, and keep the change only if
the improvement is larger than the seed-to-seed noise. All decisions are made on the
validation period; the test period is looked at only when a phase is finished.

## Phase A: measure reliably

- [x] **1. Ten seeds with confidence intervals.** Run seeds in parallel; report the
  interquartile mean and bootstrap confidence intervals (`rliable`). Add a script that
  compares two runs. *Why first: without this we cannot tell a real improvement from luck.*
- [x] **2. Steadier checkpoint selection.** Pick the checkpoint on a smoothed validation
  score instead of a single lucky evaluation. *Done: every run reports three rules,
  "last" (final model, unbiased, the headline number), "smoothed" and "best".*

## Phase B: fix the agent's behaviour

- [x] **3. Stop the overtrading.** Tried a switching penalty in the reward (0.002,
  0.005, 0.01) and a minimum holding period (5, 10 days). *Result (validation, "last"
  rule, 10 seeds): the penalty cut trades only from 286 to 211 with no clear gain in
  return. A 10-day minimum hold cut trades to 56 and moved log return from -0.576 to
  +0.047, a clear improvement. Kept: `--min-hold 10`. Caveat: random agents also improve
  under the same rule (-0.433 to -0.207), so most of the gain is fees saved, not skill.*
- [ ] **4. Reduce overfitting.** Smaller network, weight decay, layer normalisation,
  shorter training, noise added to features. *Target: validation score stops getting
  worse during training.*
- [ ] **5. Reward design.** Compare plain log return with a differential Sharpe reward
  and a drawdown-penalised reward. *Target: lower drawdown than buy-and-hold.*
- [ ] **6. DQN upgrades.** Dueling network, n-step returns, prioritised replay, added
  one at a time so each one's effect is measured.

## Phase C: more data, harder tests

- [ ] **7. Train on several assets.** ETH, SOL, and stock index ETFs alongside BTC, with
  one agent learning from all of them. *The strongest cure for overfitting is more data.*
- [ ] **8. Walk-forward testing.** Retrain on rolling windows and test on the following
  year, repeated across the whole history, instead of one fixed split.

## Phase D: stronger comparisons

- [ ] **9. A second algorithm: PPO** (Stable-Baselines3), then a recurrent version that
  has memory.
- [ ] **10. A supervised baseline.** LightGBM predicting next-day return and trading on a
  threshold. *If RL cannot beat this, that is itself a finding.*
- [ ] **11. Robustness checks.** Results at several fee levels, split by market regime
  (bull / bear / sideways), and the Deflated Sharpe Ratio to account for the number of
  variants tried.

## Phase E: write it up

- [ ] **12. Medium article.** The 2023 bugs, the rebuild, and the honest results.
  Possible after Phase B or C.
- [ ] **13. Research paper.** Pick one question (reward design, the reproducibility
  study, or offline vs. online RL), run the full comparison, write it up for arXiv and
  a workshop.
