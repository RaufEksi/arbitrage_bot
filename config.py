"""
config.py -- Central configuration for the HFT OFI RL Trading Bot.

All tunable parameters, file paths, and constants are defined here.
Other modules import from this file instead of using hardcoded values.
"""

import logging

# =============================================================
# PATHS
# =============================================================
MODELS_DIR = "./models/"
LOGS_DIR = "./logs/"
DATA_DIR = "./data/"
DATA_CSV_FILENAME = "btcusdt_ofi_data.csv"
DATA_PATH = f"{DATA_DIR}{DATA_CSV_FILENAME}"

# =============================================================
# BINANCE API
# =============================================================
SYMBOL = "btcusdt"
BINANCE_REST_URL = "https://api.binance.com/api/v3/depth"
BINANCE_WS_DEPTH_URL = "wss://stream.binance.com:9443/ws/{}@depth@100ms"
BINANCE_WS_BOOKTICKER_URL = "wss://stream.binance.com:9443/ws/{}@bookTicker"
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
COMMISSION_RATE = 0.0004
STOP_LOSS_THRESHOLD = -0.05
DEFAULT_MAX_STEPS = 10_000

# Reward tuning
OVERTRADE_WINDOW = 50
OVERTRADE_MAX = 20
OVERTRADE_PENALTY = 0.0003
REDUNDANT_PENALTY = 0.0003

# State space
OFI_LOOKBACK = 5
EMA_SPAN = 20
OBS_DIM = 12

# =============================================================
# RL HYPERPARAMETERS (PPO)
# =============================================================
LEARNING_RATE = 1e-4
N_STEPS = 4096
BATCH_SIZE = 128
N_EPOCHS = 10
GAMMA = 0.95
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.05
CLIP_OBS = 10.0

TOTAL_TIMESTEPS_LOCAL = 20_000
TOTAL_TIMESTEPS_COLAB = 2_000_000
EVAL_FREQ = 10_000
TRAIN_TEST_SPLIT = 0.8

# =============================================================
# BACKTEST
# =============================================================
ANNUALIZATION_FACTOR = 252 * 24 * 60
SYNTHETIC_BACKTEST_STEPS = 5_000

# =============================================================
# LOGGING
# =============================================================
LOG_LEVEL = logging.INFO
