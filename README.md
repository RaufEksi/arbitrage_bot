# 📈 HFT Order Flow Imbalance (OFI) AI Bot

A **High-Frequency Trading (HFT)** algorithmic engine designed for Binance Futures. This project uses pure asynchronous Level-2 Order Book streaming and advanced **Machine Learning** models to predict short-term directional price volatility based on **Order Flow Imbalance (OFI)**, **Order Book Imbalance (OBI)**, and real-time **Z-Scores**.

## 🚀 The Architecture Evolution

Initially conceived as an OpenAI Gym **Reinforcement Learning (RL / PPO)** environment, the agent struggled with "Adverse Selection" and High-Frequency Taker-Fees, leading to financial death spirals despite zero-latency optimization.

**The Pivot (V8 & V9):**  
We transitioned the brain to an **Asynchronous Supervised Learning** model using **XGBoost (Hist Gradient Boosting)**.
The results on out-of-sample data were astonishing:
- Evaluated over **200,000 continuous tick events.**
- **86% - 90% Directional Recall** on true 1.0 USDT price breakdowns/breakouts.
- Model processes a continuous 150-tick sliding window in under `<1ms`.
- Fully integrated into **Binance Futures Testnet** for live predictive execution!

---

## ⚙️ Core Technical Features

- **Asynchronous Live Pipeline (`live_trader.py`)**: Uses `asyncio` and `websockets` for ultra-low latency ingestion of the `bookTicker` stream.
- **Micro-Market Feature Engineering (`xgb_hft_model.py`)**: Vectorized calculations of 20-tick EMA and 100-tick rolling Z-Scores to completely filter out market noise and expose raw liquidity shifts.
- **Server Time Drift Synchronization**: Automatically computes the local client drift against Binance's HTTP clocks (`serverTime`) and adjusts real-time `Timestamp` variables, eliminating Binance `Error -1021`.
- **Built-in PnL & Position Tracking**: The Live Engine logs internal state, tracks double-order flipping (Short to Long), and auto-deducts the 0.04% commission fee for precise profit monitoring.
- **Aggression Cooldowns**: Implements strict 50-tick cooldowns post-execution to prevent spamming rate-limits during sustained volatility spikes.

---

## 🛠️ Usage & Installation

### 1. Requirements & Setup
Clone the repo and activate your virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
*(Dependencies: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`, `asyncio`, `websockets`, `aiohttp`)*

### 2. Model Training & Backtesting
Process the historical HFT Level-2 Data and train the XGBoost engine:
```bash
python xgb_hft_model.py
```
*This splits data chronologically (80/20) preventing temporal leakage, logs a detailed classification report, saves the optimal tree structures to `models/xgb_best.json`, and outputs a vectorized Equity Curve.*

### 3. Live Testnet Trading
Enter your Binance Testnet API keys in `config.py`:
```python
TESTNET_API_KEY = "YOUR_TESTNET_API_KEY"
TESTNET_API_SECRET = "YOUR_TESTNET_API_SECRET"
```

Fire up the Asynchronous Live Trader:
```bash
python live_trader.py
```
Sit back and observe the terminal log `[AL / SAT SİNYALİ]` directly triggering Binance API market orders in real-time alongside Live PnL tracking!

---

## ⚠️ Disclaimer
This software is provided for **educational and research purposes only.** High-frequency trading carries severe financial risk, especially in cryptocurrency derivatives. It is strictly recommended to run this within the Binance Testnet environment.
