"""
config.py -- Central configuration for the HFT OFI Trading Bot.

All tunable parameters, file paths, and constants are defined here.
Other modules import from this file instead of using hardcoded values.
"""

import os
import logging

# =============================================================
# .env FILE SUPPORT (optional)
# =============================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; environment variables still work

# =============================================================
# PATHS
# =============================================================
MODELS_DIR = "./models/"
LOGS_DIR = "./logs/"
DATA_DIR = "./data/"
DATA_CSV_FILENAME = "btcusdt_ofi_data.csv"
DATA_PATH = f"{DATA_DIR}{DATA_CSV_FILENAME}"

# =============================================================
# BINANCE API & TESTNET
# =============================================================
SYMBOL = "btcusdt"
BINANCE_REST_URL = "https://api.binance.com/api/v3/depth"
BINANCE_WS_DEPTH_URL = "wss://stream.binance.com:9443/ws/{}@depth@100ms"
BINANCE_WS_BOOKTICKER_URL = "wss://stream.binance.com:9443/ws/{}@bookTicker"

# API credentials are read from environment variables for security.
# Copy .env.example to .env and fill in your keys, or export them directly.
TESTNET_API_KEY = os.getenv("TESTNET_API_KEY", "")
TESTNET_API_SECRET = os.getenv("TESTNET_API_SECRET", "")
TESTNET_REST_URL = "https://testnet.binancefuture.com"
TESTNET_WSS_URL = "wss://stream.binancefuture.com/ws/{}@bookTicker"

SNAPSHOT_LIMIT = 1000
DISPLAY_LEVELS = 5

# =============================================================
# DATA COLLECTOR
# =============================================================
DEFAULT_TICKS = 10_000
SAVE_INTERVAL = 10_000
RECONNECT_DELAY_SEC = 3
MAX_RECONNECT_ATTEMPTS = 10
TPS_LOG_INTERVAL_SEC = 5

# =============================================================
# TRADING ENVIRONMENT (OFITradingEnv)
# =============================================================
# Commission is set to 0.0 for the RL environment intentionally:
# The reward function uses differential PnL and separate penalty
# mechanisms instead of raw commission deduction. See POST_MORTEM.md
# for the analysis of why commission makes HFT unviable at L1.
COMMISSION_RATE = 0.000
STOP_LOSS_THRESHOLD = -0.05
DEFAULT_MAX_STEPS = 10_000

# Reward tuning
OVERTRADE_WINDOW = 50
OVERTRADE_MAX = 20
OVERTRADE_PENALTY = 0.00103
REDUNDANT_PENALTY = 0.0008
NOISE_TRADING_PENALTY = 0.0005  # Penalty for trading in Z-score deadzone

# State space
OFI_LOOKBACK = 5
EMA_SPAN = 20
OBS_DIM = 16
VOLATILITY_WINDOW = 100

# =============================================================
# RL HYPERPARAMETERS (PPO) — Tuned via Optuna
# =============================================================
LEARNING_RATE = 0.0004843513577922506
N_STEPS = 1024
BATCH_SIZE = 256
N_EPOCHS = 4
GAMMA = 0.9892420260354073
GAE_LAMBDA = 0.8429234126788765
CLIP_RANGE = 0.3
ENT_COEF = 0.020901511550742748
CLIP_OBS = 10.0

TOTAL_TIMESTEPS_LOCAL = 20_000
TOTAL_TIMESTEPS_COLAB = 2_000_000
EVAL_FREQ = 50_000
EVAL_EPISODES = 1
EVAL_MAX_STEPS = 5_000
N_ENVS = 8  # Parallel environments for SubprocVecEnv
TRAIN_TEST_SPLIT = 0.8

# =============================================================
# BACKTEST
# =============================================================
ANNUALIZATION_FACTOR = 252 * 24 * 60
SYNTHETIC_BACKTEST_STEPS = 5_000

# =============================================================
# LIVE TRADER
# =============================================================
WINDOW_SIZE = 150       # Rolling window required for 100-tick Z-scores
COOLDOWN_TICKS = 50     # Rate-limit: wait N ticks after an execution
TRADE_QTY = 0.001       # BTC quantity per trade (risk management)
TAKER_COMMISSION = 0.0004  # Binance Futures taker fee

# =============================================================
# LOGGING
# =============================================================
LOG_LEVEL = logging.INFO
