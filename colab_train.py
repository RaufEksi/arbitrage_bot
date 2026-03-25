# =============================================================
# HFT OFI Trading Bot -- Google Colab Training Script (V4)
# =============================================================
# KULLANIM:
# 1. Google Colab > Runtime > Change runtime type > GPU
# 2. Bu dosyanin icerigini tek bir hucreye yapistirin
# 3. Ilk olarak env.py, config.py ve logger.py dosyalari yuklenecek
# 4. Ardindan CSV yukleme penceresi acilir -> btcusdt_ofi_data.csv yukleyin
# 5. Egitim bittikten sonra best_model.zip + vec_normalize.pkl indirilir
# 6. Indirilen dosyalari projenizin models/ klasorune koyun
# =============================================================

# --- CELL 1: Kurulum ---
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "stable-baselines3[extra]", "gymnasium", "numpy",
                       "pandas", "tensorboard"])

# --- CELL 2: Proje dosyalarini yukle (DRY - tek kaynak) ---
from google.colab import files
import os

print("="*60)
print("  PROJE DOSYALARINI YUKLEYIN")
print("  Lutfen su 3 dosyayi secin: env.py, config.py, logger.py")
print("="*60 + "\n")

uploaded_src = files.upload()
print(f"\nYuklenen dosyalar: {list(uploaded_src.keys())}")

# Dogrulama
for required_file in ["env.py", "config.py", "logger.py"]:
    if not os.path.exists(required_file):
        raise FileNotFoundError(f"{required_file} yuklenmedi! Lutfen tekrar deneyin.")
print("Tum kaynak dosyalar yuklendi.")

# --- CELL 3: CSV yukle ---
import pandas as pd

print("\n" + "="*60)
print("  GERCEK VERI YUKLEME")
print("  Lutfen btcusdt_ofi_data.csv dosyanizi secin.")
print("="*60 + "\n")

uploaded_csv = files.upload()
csv_filename = list(uploaded_csv.keys())[0]
full_df = pd.read_csv(csv_filename)
print(f"Toplam satir: {len(full_df):,}")

required_cols = {"bid_price", "ask_price", "ofi", "spread"}
missing = required_cols - set(full_df.columns)
if missing:
    raise ValueError(f"CSV'de eksik kolonlar: {missing}")

# --- CELL 4: Import ve egitim ---
import numpy as np
import config as cfg
from env import OFITradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

MODELS_DIR = cfg.MODELS_DIR
LOGS_DIR = cfg.LOGS_DIR
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Train/Test Split
split_idx = int(len(full_df) * cfg.TRAIN_TEST_SPLIT)
train_df = full_df.iloc[:split_idx].reset_index(drop=True)
test_df = full_df.iloc[split_idx:].reset_index(drop=True)
print(f"Train: {len(train_df):,} satir | Test: {len(test_df):,} satir")

def make_train_env():
    return OFITradingEnv(df=train_df)

def make_eval_env():
    eval_slice = test_df.iloc[:cfg.EVAL_MAX_STEPS].reset_index(drop=True)
    return Monitor(OFITradingEnv(df=eval_slice))

env = DummyVecEnv([make_train_env])
eval_env = DummyVecEnv([make_eval_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=cfg.CLIP_OBS)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=cfg.CLIP_OBS)

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=MODELS_DIR,
    log_path=LOGS_DIR, 
    eval_freq=cfg.EVAL_FREQ,
    n_eval_episodes=cfg.EVAL_EPISODES,
    deterministic=True, 
    render=False
)

TOTAL_TIMESTEPS = cfg.TOTAL_TIMESTEPS_COLAB

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=cfg.LEARNING_RATE,
    n_steps=cfg.N_STEPS,
    batch_size=cfg.BATCH_SIZE,
    n_epochs=cfg.N_EPOCHS,
    gamma=cfg.GAMMA,
    gae_lambda=cfg.GAE_LAMBDA,
    clip_range=cfg.CLIP_RANGE,
    ent_coef=cfg.ENT_COEF,
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

# --- CELL 5: Indir ---
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
