# HFT Order Flow Imbalance (OFI) Reinforcement Learning Trading Bot

A high-frequency cryptocurrency trading system that uses Level 2 Order Book data to compute Order Flow Imbalance (OFI) and trains a Reinforcement Learning agent to make trading decisions based on market microstructure signals.

## Overview

This project implements the OFI strategy described by Cont et al. (2014) within a custom Gymnasium environment. A PPO (Proximal Policy Optimization) agent is trained to interpret order flow dynamics and execute trades while managing transaction costs, slippage, and position risk.

Unlike traditional price-prediction models, this system focuses on **liquidity flow** and **market microstructure** to identify short-term trading opportunities.

## Architecture

```
binance_ofi_bot/
├── config.py              # API endpoints, symbols, WebSocket parameters
├── logger.py              # Loguru-based structured logging
├── orderbook.py           # L2 Order Book synchronization (REST + WS)
├── ofi_calculator.py      # OFI computation (Cont et al. 2014)
├── websocket_manager.py   # Asynchronous WebSocket client (asyncio)
├── main.py                # Main controller / orchestrator
├── env.py                 # Custom Gymnasium trading environment (V2)
├── train_agent.py         # PPO training pipeline with VecNormalize
├── backtest.py            # Backtesting engine with financial metrics
├── colab_train.py         # Self-contained Google Colab training script
├── test_env.py            # Environment unit tests
└── requirements.txt       # Python dependencies
```

## Environment Design (V2)

### Observation Space (12 dimensions)

| Index | Feature              | Purpose                                      |
|-------|----------------------|----------------------------------------------|
| 0-4   | OFI lookback (5 ticks) | Captures short-term order flow trend         |
| 5     | OFI EMA-20           | Smoothed signal for noise filtering           |
| 6     | Spread               | Current liquidity cost                        |
| 7     | Spread delta         | Volatility and liquidity change detection     |
| 8     | Position             | Current position state (-1, 0, 1)             |
| 9     | Unrealized PnL       | Open position profit/loss                     |
| 10    | Holding time         | Normalized duration of current position       |
| 11    | Step progress        | Episode completion ratio                      |

### Action Space (Discrete)

| Action | Description                                |
|--------|--------------------------------------------|
| 0      | Hold -- maintain current position          |
| 1      | Buy -- go long or close short              |
| 2      | Sell -- go short or close long             |

### Reward Mechanism

The reward function is designed to prevent common RL pitfalls in financial environments:

- **Differential PnL**: Rewards the *change* in unrealized PnL per tick, not the cumulative value, preventing double-counting exploits.
- **Transaction Costs**: Commission and slippage are deducted on every trade execution.
- **Realized PnL**: Credited upon closing a position.
- **Hold Bonus**: Small positive reward for maintaining a profitable position, incentivizing patience.
- **Flat Bonus**: Reward for staying flat when the OFI signal is weak, discouraging noise trading.
- **Overtrading Penalty**: Penalizes excessive trading frequency within a rolling window.
- **Redundant Action Penalty**: Penalizes no-op actions (e.g., selling while already short).

## Installation

```bash
# Clone the repository
git clone https://github.com/RaufEksi/arbitrage_bot.git
cd arbitrage_bot

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
# source venv/bin/activate       # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_agent.py
```

The training script uses PPO with the following V2 hyperparameters optimized for financial time series:

| Parameter       | Value   | Rationale                                    |
|-----------------|---------|----------------------------------------------|
| `learning_rate` | 1e-4    | Slower convergence for noisy financial data  |
| `n_steps`       | 4096    | Longer rollouts for stable gradient updates  |
| `batch_size`    | 128     | Larger batches for 12-dim observation space  |
| `gamma`         | 0.95    | Short-term focus appropriate for HFT         |
| `ent_coef`      | 0.02    | Higher exploration to avoid premature convergence |

Models are saved to `models/best_model.zip` and normalization statistics to `models/vec_normalize.pkl`.

### Training on Google Colab

For faster training with GPU acceleration, use the self-contained Colab script:

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. Set runtime to GPU: `Runtime > Change runtime type > T4 GPU`.
3. Paste the contents of `colab_train.py` into a cell and run.
4. Download the generated `best_model.zip` and `vec_normalize.pkl` files.
5. Place both files in the `models/` directory of your local project.

### Backtesting

```bash
python backtest.py
```

The backtesting engine runs the trained agent deterministically over synthetic data and reports:

- Action distribution (Hold / Buy / Sell counts)
- Net profit/loss
- Annualized Sharpe Ratio (using `sqrt(252 * 24 * 60)` for crypto HFT)
- Equity curve visualization saved to `equity_curve.png`

### Live Data Collection

```bash
python main.py
```

Connects to the Binance WebSocket API and streams real-time L2 order book data for OFI calculation.

### Monitoring

```bash
tensorboard --logdir=logs/
```

## Technical Stack

| Component          | Technology                          |
|--------------------|-------------------------------------|
| Data Source         | Binance L2 WebSocket API (asyncio) |
| RL Environment      | Gymnasium (custom `OFITradingEnv`) |
| Agent Algorithm     | PPO via Stable-Baselines3          |
| Observation Scaling | VecNormalize                       |
| Logging             | TensorBoard, Loguru                |
| Data Processing     | NumPy, Pandas                      |
| Visualization       | Matplotlib                         |

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events.* Journal of Financial Econometrics, 12(1), 47-88.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.

## License

This project is provided for educational and research purposes only. It does not constitute financial advice. Use at your own risk.
