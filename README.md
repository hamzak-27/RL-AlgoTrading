# rltrader

Deep reinforcement learning for single-asset trading, rebuilt from a 2023 notebook
(kept in `legacy/`) with the aim of honest, reproducible results.

## Setup

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -e .[dev]
```

## Run

```bash
.venv/Scripts/python scripts/download_data.py BTC-USD   # daily bars -> data/BTC-USD.csv
.venv/Scripts/python -m pytest                          # check the simulator
.venv/Scripts/python scripts/train.py --seeds 5         # train + evaluate -> results/BTC-USD/
```

## Layout

| File | What it does |
|---|---|
| `src/rltrader/data.py` | Download/load bars (always oldest to newest), chronological train/val/test split |
| `src/rltrader/features.py` | Backward-looking, scaled features (returns, volatility, trend, RSI, volume) |
| `src/rltrader/env.py` | Gymnasium environment: target-position actions, next-open execution, fees, log-return reward |
| `src/rltrader/dqn.py` | Double DQN with replay buffer and target network |
| `src/rltrader/evaluate.py` | Episode runner, performance metrics, baselines |
| `scripts/train.py` | Multi-seed training, validation checkpointing, test report and chart |
| `tests/test_env.py` | Accounting and no-look-ahead checks |

## Periods

| Period | Dates | Used for |
|---|---|---|
| train | start to 2021-12-31 | learning |
| val | 2022-01-01 to 2023-12-31 | choosing the best checkpoint |
| test | 2024-01-01 onwards | final reported numbers only |
