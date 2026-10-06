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
- [x] **4. Reduce overfitting.** Tried a smaller network, weight decay, layer
  normalisation, state noise, shorter training and a shorter input window, each on top
  of the 10-day hold. *Result (validation, "last" rule, 10 seeds): none clearly beat the
  baseline on return. The overfitting gap (edge over buy-and-hold per year, train minus
  validation) was 0.40 for the baseline; a 3-day window cut it to 0.08 with validation
  return no worse (+0.137 vs +0.047, not a clear difference). Kept: `--window 3`, as the
  simpler model. Conclusion: regularisation does not create skill here; more data
  (item 7) is the likelier cure.*
- [x] **5. Reward design.** (Done after item 7, on all 11 assets.) Compared plain log
  return with volatility-scaled return, differential Sharpe, a drawdown penalty and
  double-counted losses. *Result ("last" rule, 10 seeds, all-asset validation): no reward
  clearly raised the edge over random (baseline +0.129; vol-scaled +0.160; differential
  Sharpe +0.108; drawdown +0.068; downside +0.092). Vol-scaled clearly cut trades (52 to
  43 per asset). Downside clearly cut max drawdown (-46% to -41%) but by holding less
  (39% of days vs 55%) and earning less. Drawdown penalty clearly lowered Sharpe. Kept:
  `--reward vol_scaled`.*
- [x] **6. DQN upgrades.** Dueling network, 3-step returns, prioritised replay, and all
  three together, on the item 5 setup. *Result ("last" rule, 10 seeds, all-asset
  validation, edge over random): baseline +0.160; dueling +0.128; 3-step +0.111;
  prioritised +0.079; all three +0.112. None helped; 3-step clearly raised trading (43 to
  56 per asset). Kept: none.*

**Phase B final setup:** train on 11 assets, 10-day minimum hold, 3-day window,
volatility-scaled reward, plain Double DQN (run name `rw_vol_scaled`).

**Phase B test result** (2024-01 to 2026-10, all 11 assets, 10 seeds, log return per
year, looked at once): DQN +0.067; random -0.015; SMA 20/50 +0.120; buy-and-hold +0.116.
Edge over random +0.068 [+0.046, +0.116] (9 of 10 seeds positive); edge over SMA -0.066
[-0.089, -0.019]; edge over buy-and-hold -0.063 [-0.086, -0.015]. The agent beats
no-skill trading and loses to both simple baselines.

## Phase C: more data, harder tests

- [x] **7. Train on several assets.** (Done before items 5 and 6, because item 4
  pointed to lack of data.) Trained on 8 crypto assets, and on 8 crypto plus SPY, QQQ and
  GLD; scored every run on the validation period of all 11 assets against 100 random
  agents under the same rules. *Result ("last" rule, 10 seeds, edge over random in log
  return per year): BTC-only +0.110 [+0.034, +0.184]; 8 crypto +0.104; all 11 +0.129
  [+0.091, +0.175]; 8 crypto with 150k steps +0.117 [+0.093, +0.150]. No clear gain in
  the average, but the spread across seeds roughly halved and every seed was positive.
  A plain SMA 20/50 rule scores +0.088 on the same measure with a third of the trades.
  Kept: training on all 11 assets, and all-asset scoring as the headline measurement.*
- [x] **8. Walk-forward testing.** Retrained the Phase B setup for each year 2019 to
  2026 on all earlier data and tested on that year (10 seeds, up to 11 assets, log return
  per year). *Result: edge over random +0.156 [+0.134, +0.190], positive in 7 of 8 years;
  edge over SMA 20/50 -0.033 [-0.055, +0.001], ahead in 3 of 8 years; edge over
  buy-and-hold -0.038 [-0.061, -0.005]. The agent reliably beats no-skill trading and
  roughly ties the simple baselines. Caveat: settings were chosen on 2022-2023 data.*

## Phase D: stronger comparisons

- [ ] **9. A second algorithm: PPO** (Stable-Baselines3), then a recurrent version that
  has memory.
- [x] **10. A supervised baseline.** LightGBM (untuned) predicting the 10-day forward
  return from the same features, trading long when the prediction is positive, through
  the same walk-forward folds. *Result: edge over random +0.174 [+0.167, +0.197], positive
  in 8 of 8 years; edge over SMA -0.015 [-0.022, +0.008]. Against the DQN: +0.018
  [-0.015, +0.050], not a clear difference, with clearly fewer trades and a much tighter
  spread across seeds. RL adds nothing over the supervised model here.*
- [ ] **11. Robustness checks.** Results at several fee levels, split by market regime
  (bull / bear / sideways), and the Deflated Sharpe Ratio to account for the number of
  variants tried.

## Phase E: write it up

- [ ] **12. Medium article.** The 2023 bugs, the rebuild, and the honest results.
  Possible after Phase B or C.
- [ ] **13. Research paper.** Pick one question (reward design, the reproducibility
  study, or offline vs. online RL), run the full comparison, write it up for arXiv and
  a workshop.
