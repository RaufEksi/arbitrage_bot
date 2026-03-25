# =============================================================
# HFT OFI Trading Bot -- Google Colab Training Script (V4)
# =============================================================
# SubprocVecEnv ile paralel CPU + GPU egitimi
#
# KULLANIM:
# 1. Google Colab > Runtime > Change runtime type > GPU (A100)
# 2. Bu dosyanin icerigini tek bir hucreye yapistirin
# 3. Ilk upload: env.py, config.py, logger.py
# 4. Ikinci upload: btcusdt_ofi_data.csv
# 5. Egitim bittikten sonra best_model.zip + vec_normalize.pkl indirilir
# =============================================================

# --- CELL 1: Kurulum ---
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "stable-baselines3[extra]", "gymnasium", "numpy",
                       "pandas", "tensorboard"])

# --- CELL 2: Proje dosyalarini yukle ---
from google.colab import files
import os

print("="*60)
print("  PROJE DOSYALARINI YUKLEYIN")
print("  Lutfen su 3 dosyayi secin: env.py, config.py, logger.py")
print("="*60 + "\n")

uploaded_src = files.upload()
for f in ["env.py", "config.py", "logger.py"]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f} yuklenmedi!")
print("Kaynak dosyalar yuklendi.")

# --- CELL 3: CSV yukle ---
import pandas as pd

print("\n" + "="*60)
print("  GERCEK VERI YUKLEME")
print("  Lutfen btcusdt_ofi_data.csv dosyanizi secin.")
print("="*60 + "\n")

uploaded_csv = files.upload()
csv_filename = list(uploaded_csv.keys())[0]

# Save with standard name so SubprocVecEnv workers can find it
STANDARD_CSV = "btcusdt_ofi_data.csv"
if csv_filename != STANDARD_CSV:
    os.rename(csv_filename, STANDARD_CSV)
    csv_filename = STANDARD_CSV

full_df = pd.read_csv(csv_filename)
print(f"Toplam satir: {len(full_df):,}")

required_cols = {"bid_price", "ask_price", "ofi", "spread"}
missing = required_cols - set(full_df.columns)
if missing:
    raise ValueError(f"Eksik kolonlar: {missing}")

# --- CELL 4: config.py DATA_PATH override for Colab ---
import config as cfg

# Override DATA_PATH to point to uploaded CSV in Colab's working dir
cfg.DATA_PATH = STANDARD_CSV

# --- CELL 5: Paralel egitim ---
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from env import OFITradingEnv

MODELS_DIR = cfg.MODELS_DIR
LOGS_DIR = cfg.LOGS_DIR
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Train/Test Split
split_idx = int(len(full_df) * cfg.TRAIN_TEST_SPLIT)
train_df = full_df.iloc[:split_idx].reset_index(drop=True)
test_df = full_df.iloc[split_idx:].reset_index(drop=True)
n_envs = cfg.N_ENVS
print(f"Train: {len(train_df):,} | Test: {len(test_df):,} | Envs: {n_envs}")

# -- Picklable factories: each worker loads CSV from disk --
def _make_train_env(rank):
    def _init():
        df = pd.read_csv(cfg.DATA_PATH)
        si = int(len(df) * cfg.TRAIN_TEST_SPLIT)
        return OFITradingEnv(df=df.iloc[:si].reset_index(drop=True))
    return _init

def _make_eval_env():
    def _init():
        df = pd.read_csv(cfg.DATA_PATH)
        si = int(len(df) * cfg.TRAIN_TEST_SPLIT)
        test = df.iloc[si:].reset_index(drop=True)
        return Monitor(OFITradingEnv(df=test.iloc[:cfg.EVAL_MAX_STEPS].reset_index(drop=True)))
    return _init

# Parallel train envs, single eval env
env = SubprocVecEnv([_make_train_env(i) for i in range(n_envs)])
eval_env = DummyVecEnv([_make_eval_env()])

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=cfg.CLIP_OBS)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=cfg.CLIP_OBS)

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=MODELS_DIR,
    log_path=LOGS_DIR, 
    eval_freq=max(cfg.EVAL_FREQ // n_envs, 1),
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

print(f"\nV4 Egitim: {TOTAL_TIMESTEPS:,} timestep | {n_envs} paralel ortam | GPU: {model.device}")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    tb_log_name="PPO_OFI_V4"
)

model.save(os.path.join(MODELS_DIR, "final_trading_model"))
vec_norm_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")
env.save(vec_norm_path)
env.close()
eval_env.close()
print("V4 Egitim tamamlandi!")

# --- CELL 6: Indir ---
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

print("\nIndirilen dosyalari models/ klasorune koyun:")
print("  python backtest.py")
