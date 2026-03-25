# =============================================================
# HFT OFI Trading Bot -- Google Colab Training Script (V4)
# =============================================================
# SubprocVecEnv ile paralel CPU + GPU egitimi
#
# KULLANIM:
# 1. Google Colab > Runtime > Change runtime type > GPU (A100)
# 2. Bu dosyanin icerigini tek bir hucreye yapistirin
# 3. Sol panelden (dosya ikonu) env.py, config.py, logger.py yukleyin
# 4. Sol panelden btcusdt_ofi_data.csv yukleyin
# 5. Hucreyi calistirin
# =============================================================

# --- CELL 1: Kurulum ---
import subprocess, sys, os
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "stable-baselines3[extra]", "gymnasium", "numpy",
                       "pandas", "tensorboard"])

# --- CELL 2: Dosya kontrolu ---
# Sol panelden (klasor ikonu) dosyalari Colab'a surukle-birak ile yukleyin.
# Dosyalar /content/ dizininde olmalidir.

REQUIRED_SRC  = ["env.py", "config.py", "logger.py"]
CSV_FILENAME  = "btcusdt_ofi_data.csv"

print("="*60)
print("  DOSYA KONTROLU")
print("="*60)

missing = [f for f in REQUIRED_SRC if not os.path.exists(f)]
if missing:
    print(f"\n  EKSIK DOSYALAR: {missing}")
    print("  Sol panelden (klasor ikonu) bu dosyalari yukleyin.")
    print("  Sonra bu hucreyi tekrar calistirin.\n")
    raise FileNotFoundError(f"Eksik: {missing}")

if not os.path.exists(CSV_FILENAME):
    print(f"\n  EKSIK: {CSV_FILENAME}")
    print("  Sol panelden CSV dosyasini yukleyin.")
    print("  Sonra bu hucreyi tekrar calistirin.\n")
    raise FileNotFoundError(f"Eksik: {CSV_FILENAME}")

print("  env.py         ✓")
print("  config.py      ✓")
print("  logger.py      ✓")
print(f"  {CSV_FILENAME} ✓")
print("="*60)

# --- CELL 3: Veri yukle ve kontrol et ---
import pandas as pd

full_df = pd.read_csv(CSV_FILENAME)
print(f"\nToplam satir: {len(full_df):,}")

required_cols = {"bid_price", "ask_price", "ofi", "spread"}
missing_cols = required_cols - set(full_df.columns)
if missing_cols:
    raise ValueError(f"CSV'de eksik kolonlar: {missing_cols}")
print("Kolon kontrolu GECTI ✓")

# --- CELL 4: Config ve ortam ---
import numpy as np
import config as cfg
from env import OFITradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Override DATA_PATH -> Colab working dir
cfg.DATA_PATH = CSV_FILENAME

MODELS_DIR = cfg.MODELS_DIR
LOGS_DIR = cfg.LOGS_DIR
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Train/Test Split
split_idx = int(len(full_df) * cfg.TRAIN_TEST_SPLIT)
train_df = full_df.iloc[:split_idx].reset_index(drop=True)
test_df = full_df.iloc[split_idx:].reset_index(drop=True)

n_envs = getattr(cfg, 'N_ENVS', 8)
eval_max = getattr(cfg, 'EVAL_MAX_STEPS', 5000)
eval_eps = getattr(cfg, 'EVAL_EPISODES', 1)
eval_freq = getattr(cfg, 'EVAL_FREQ', 50000)

print(f"Train: {len(train_df):,} | Test: {len(test_df):,} | Envs: {n_envs}")

# --- CELL 5: Paralel ortam ve egitim ---
# DummyVecEnv with multiple envs (SubprocVecEnv crashes in Colab notebooks)
# Numpy-optimized env.py ensures high FPS even without multiprocessing.

def _make_train_env(train_data):
    """Creates a train env from in-memory DataFrame."""
    def _init():
        return OFITradingEnv(df=train_data)
    return _init

def _make_eval_env(test_data, max_steps):
    """Creates an eval env from in-memory DataFrame."""
    def _init():
        return Monitor(OFITradingEnv(df=test_data.iloc[:max_steps].reset_index(drop=True)))
    return _init

env = DummyVecEnv([_make_train_env(train_df) for _ in range(n_envs)])
eval_env = DummyVecEnv([_make_eval_env(test_df, eval_max)])

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=cfg.CLIP_OBS)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=cfg.CLIP_OBS)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODELS_DIR,
    log_path=LOGS_DIR,
    eval_freq=max(eval_freq // n_envs, 1),
    n_eval_episodes=eval_eps,
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

print(f"\nV4 Egitim Basliyor")
print(f"  Timestep  : {TOTAL_TIMESTEPS:,}")
print(f"  Envs      : {n_envs} paralel")
print(f"  GPU       : {model.device}")
print(f"  Eval freq : her {eval_freq:,} step\n")

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
print("\nV4 Egitim tamamlandi!")

# --- CELL 6: Indir ---
from google.colab import files

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
