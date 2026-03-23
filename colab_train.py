# =============================================================
# HFT OFI Trading Bot -- Google Colab Training Script (V4)
# =============================================================
# KULLANIM:
# 1. Google Colab > Runtime > Change runtime type > GPU
# 2. Bu dosyanin icerigini tek bir hucreye yapistirin
# 3. CSV yukleme penceresi acilinca btcusdt_ofi_data.csv yukleyin
# 4. Egitim bittikten sonra best_model.zip + vec_normalize.pkl indirilir
# 5. Indirilen 2 dosyayi projenizin models/ klasorune koyun
# =============================================================

# --- CELL 1: Kurulum ---
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "stable-baselines3[extra]", "gymnasium", "numpy",
                       "pandas", "tensorboard"])

# --- CELL 2: V4 env.py (bonuslar kaldirildi, cezalar yumusaklandi) ---
ENV_CODE = r'''
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger("OFITradingEnv")
logging.basicConfig(level=logging.WARNING)

class OFITradingEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}
    
    # V4: Softened penalties, no bonuses
    OVERTRADE_WINDOW = 50
    OVERTRADE_MAX = 20
    OVERTRADE_PENALTY = 0.0003
    REDUNDANT_PENALTY = 0.0003
    OFI_LOOKBACK = 5
    EMA_SPAN = 20
    
    def __init__(self, commission_rate=0.0004, render_mode=None, max_steps=10000, df=None):
        super().__init__()
        self.commission_rate = commission_rate
        self.render_mode = render_mode
        
        self.df = df
        if self.df is not None:
            required = {"bid_price", "ask_price", "ofi", "spread"}
            missing = required - set(self.df.columns)
            if missing:
                raise ValueError(f"DataFrame missing columns: {missing}")
            self.max_steps = len(self.df) - 1
            print(f"DataFrame mode: {len(self.df):,} rows, max_steps={self.max_steps}")
        else:
            self.max_steps = max_steps
        
        self.action_space = spaces.Discrete(3)
        OBS_DIM = 12
        low = np.full(OBS_DIM, -np.inf, dtype=np.float32)
        high = np.full(OBS_DIM, np.inf, dtype=np.float32)
        low[8], high[8] = -1.0, 1.0
        low[10], high[10] = 0.0, 1.0
        low[11], high[11] = 0.0, 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.ofi_history = deque(maxlen=self.OFI_LOOKBACK)
        self.ofi_ema = 0.0
        self.ema_alpha = 2.0 / (self.EMA_SPAN + 1)
        self.trade_history = deque(maxlen=self.OVERTRADE_WINDOW)
        
        self.state = np.zeros(OBS_DIM, dtype=np.float32)
        self.current_position = 0
        self.entry_price = 0.0
        self.cumulative_reward = 0.0
        self.current_step = 0
        self.prev_unrealized_pnl = 0.0
        self.holding_time = 0
        self.prev_spread = 0.0
        self.latest_ofi = 0.0
        self.latest_bid = 0.0
        self.latest_ask = 0.0
        self.latest_spread = 0.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_position = 0
        self.entry_price = 0.0
        self.cumulative_reward = 0.0
        self.current_step = 0
        self.prev_unrealized_pnl = 0.0
        self.holding_time = 0
        self.prev_spread = 0.0
        self.ofi_ema = 0.0
        self.ofi_history.clear()
        self.trade_history.clear()
        for _ in range(self.OFI_LOOKBACK):
            self.ofi_history.append(0.0)
        self.latest_ofi = 0.0
        self.latest_spread = 0.0
        self.latest_bid = 0.0
        self.latest_ask = 0.0
        self.state = np.zeros(12, dtype=np.float32)
        return self.state, {}

    def _build_state(self, unrealized_pnl):
        ofi_w = list(self.ofi_history)
        sd = (self.latest_spread - self.prev_spread) if self.prev_spread > 0 else 0.0
        ms = self.max_steps if self.max_steps > 0 else 10000
        return np.array([
            ofi_w[-5] if len(ofi_w)>=5 else 0, ofi_w[-4] if len(ofi_w)>=4 else 0,
            ofi_w[-3] if len(ofi_w)>=3 else 0, ofi_w[-2] if len(ofi_w)>=2 else 0,
            ofi_w[-1] if len(ofi_w)>=1 else 0, self.ofi_ema, self.latest_spread, sd,
            float(self.current_position), unrealized_pnl,
            min(self.holding_time/ms, 1.0), min(self.current_step/ms, 1.0)
        ], dtype=np.float32)

    def _load_tick_from_df(self):
        idx = min(self.current_step, len(self.df) - 1)
        row = self.df.iloc[idx]
        self.update_market_data(float(row["ofi"]), float(row["bid_price"]), float(row["ask_price"]))

    def step(self, action):
        self.current_step += 1
        self.holding_time += 1
        
        if self.df is not None:
            self._load_tick_from_df()
        
        prev_position = self.current_position
        step_reward = 0.0
        trade_executed = False
        realized_pnl = 0.0
        
        cur_pnl = 0.0
        if prev_position == 1 and self.entry_price > 0:
            cur_pnl = (self.latest_bid - self.entry_price) / self.entry_price
        elif prev_position == -1 and self.entry_price > 0:
            cur_pnl = (self.entry_price - self.latest_ask) / self.entry_price
        delta_pnl = cur_pnl - self.prev_unrealized_pnl
        step_reward += delta_pnl

        if action == 1:
            if self.current_position <= 0:
                ep = self.latest_ask
                if ep > 0:
                    step_reward -= (self.commission_rate + self.latest_spread/2.0/ep)
                    if self.current_position == -1 and self.entry_price > 0:
                        realized_pnl = (self.entry_price - ep) / self.entry_price
                        step_reward += realized_pnl
                    self.current_position = 1
                    self.entry_price = ep
                    trade_executed = True
                    self.prev_unrealized_pnl = 0.0
                    self.holding_time = 0
                    self.trade_history.append(1)
            else:
                step_reward -= self.REDUNDANT_PENALTY
        elif action == 2:
            if self.current_position >= 0:
                ep = self.latest_bid
                if ep > 0:
                    step_reward -= (self.commission_rate + self.latest_spread/2.0/ep)
                    if self.current_position == 1 and self.entry_price > 0:
                        realized_pnl = (ep - self.entry_price) / self.entry_price
                        step_reward += realized_pnl
                    self.current_position = -1
                    self.entry_price = ep
                    trade_executed = True
                    self.prev_unrealized_pnl = 0.0
                    self.holding_time = 0
                    self.trade_history.append(1)
            else:
                step_reward -= self.REDUNDANT_PENALTY
        else:
            self.trade_history.append(0)
        
        # V4: Only soft overtrading penalty, no bonuses
        rt = sum(self.trade_history)
        if rt > self.OVERTRADE_MAX:
            step_reward -= (rt - self.OVERTRADE_MAX) * self.OVERTRADE_PENALTY

        unrealized_pnl = 0.0
        if self.current_position == 1 and self.entry_price > 0:
            unrealized_pnl = (self.latest_bid - self.entry_price) / self.entry_price
        elif self.current_position == -1 and self.entry_price > 0:
            unrealized_pnl = (self.entry_price - self.latest_ask) / self.entry_price
        self.prev_unrealized_pnl = unrealized_pnl
        self.prev_spread = self.latest_spread
        self.state = self._build_state(unrealized_pnl)
        self.cumulative_reward += step_reward
        
        info = {"pnl": unrealized_pnl, "realized_pnl": realized_pnl,
                "position": self.current_position, "trade_executed": trade_executed,
                "reward": step_reward, "total_trades": sum(self.trade_history)}
        terminated = unrealized_pnl < -0.05
        truncated = self.max_steps > 0 and self.current_step >= self.max_steps
        return self.state, float(step_reward), terminated, truncated, info

    def update_market_data(self, ofi, bid, ask):
        self.latest_ofi = ofi
        self.latest_bid = bid
        self.latest_ask = ask
        self.latest_spread = ask - bid
        self.ofi_history.append(ofi)
        self.ofi_ema = self.ema_alpha * ofi + (1 - self.ema_alpha) * self.ofi_ema
        unrealized_pnl = 0.0
        if self.current_position == 1 and self.entry_price > 0:
            unrealized_pnl = (bid - self.entry_price) / self.entry_price
        elif self.current_position == -1 and self.entry_price > 0:
            unrealized_pnl = (self.entry_price - ask) / self.entry_price
        self.state = self._build_state(unrealized_pnl)
    
    def render(self):
        if self.render_mode == "ansi":
            print(f"Step: {self.current_step} | Pos: {self.current_position} | Reward: {self.cumulative_reward:.5f}")
'''

with open("ofi_env.py", "w") as f:
    f.write(ENV_CODE)
print("V4 ofi_env.py yazildi (bonuslar kaldirildi, cezalar 0.0003).")

# --- CELL 3: CSV YUKLE ---
import os
import pandas as pd
from google.colab import files

print("\n" + "="*60)
print("  GERCEK VERI YUKLEME")
print("  Lutfen btcusdt_ofi_data.csv dosyanizi secin.")
print("="*60 + "\n")

uploaded = files.upload()
csv_filename = list(uploaded.keys())[0]
full_df = pd.read_csv(csv_filename)
print(f"Toplam satir: {len(full_df):,}")

required = {"bid_price", "ask_price", "ofi", "spread"}
missing = required - set(full_df.columns)
if missing:
    raise ValueError(f"CSV'de eksik kolonlar: {missing}")

# 80/20 Train/Test Split
split_idx = int(len(full_df) * 0.8)
train_df = full_df.iloc[:split_idx].reset_index(drop=True)
test_df = full_df.iloc[split_idx:].reset_index(drop=True)
print(f"Train: {len(train_df):,} satir | Test: {len(test_df):,} satir")

# --- CELL 4: V4 EGITIM ---
import numpy as np
import importlib
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

import ofi_env
importlib.reload(ofi_env)
from ofi_env import OFITradingEnv

MODELS_DIR = "./models/"
LOGS_DIR = "./logs/"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Train ortami: sadece ilk %80
def make_train_env():
    return OFITradingEnv(commission_rate=0.0004, df=train_df)

# Eval ortami: son %20 (out-of-sample)
def make_eval_env():
    return OFITradingEnv(commission_rate=0.0004, df=test_df)

env = DummyVecEnv([make_train_env])
eval_env = DummyVecEnv([make_eval_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=MODELS_DIR,
    log_path=LOGS_DIR, 
    eval_freq=50000,
    deterministic=True, 
    render=False
)

# ================================================================
# V4 HYPERPARAMETERS
# ================================================================
TOTAL_TIMESTEPS = 2_000_000

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=128,
    n_epochs=10,
    gamma=0.95,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,             # V4: Higher exploration
    verbose=1,
    tensorboard_log=LOGS_DIR,
    device="auto"
)

print(f"\nV4 Egitim: {TOTAL_TIMESTEPS:,} timestep | Train: {len(train_df):,} satir")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    tb_log_name="PPO_OFI_V4"
)

model.save(os.path.join(MODELS_DIR, "final_trading_model"))
vec_norm_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")
env.save(vec_norm_path)
print("V4 Egitim tamamlandi!")

# --- CELL 5: INDIR ---
best_path = os.path.join(MODELS_DIR, "best_model.zip")
final_path = os.path.join(MODELS_DIR, "final_trading_model.zip")

if os.path.exists(best_path):
    files.download(best_path)
    print("best_model.zip indirildi!")
else:
    files.download(final_path)
    print("final_trading_model.zip indirildi!")

files.download(vec_norm_path)
print("vec_normalize.pkl indirildi!")

print("\nIndirilen dosyalari models/ klasorune koyun, sonra:")
print("  python backtest.py")
