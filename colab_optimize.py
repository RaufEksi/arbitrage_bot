# =============================================================
# HFT OFI Bot -- Optuna Hyperparameter Optimization (Colab)
# =============================================================
# KULLANIM:
# 1. Google Colab > Runtime > Change runtime type > GPU
# 2. Bu dosyanin icerigini tek bir hucreye yapistirin
# 3. Ilk upload: env.py, config.py, logger.py
# 4. Ikinci upload: btcusdt_ofi_data.csv
# 5. Optimizasyon bittikten sonra en iyi parametreler yazdirilir
# 6. Bu parametreleri config.py'a gecirin, sonra colab_train.py calistirin
# =============================================================

# --- CELL 1: Kurulum ---
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "stable-baselines3[extra]", "gymnasium", "numpy",
                       "pandas", "optuna", "tensorboard"])

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
full_df = pd.read_csv(csv_filename)
print(f"Toplam satir: {len(full_df):,}")

required_cols = {"bid_price", "ask_price", "ofi", "spread"}
missing = required_cols - set(full_df.columns)
if missing:
    raise ValueError(f"Eksik kolonlar: {missing}")

# --- CELL 4: Veri bolumu (70/10/20) ---
n = len(full_df)
train_end = int(n * 0.7)
val_end = int(n * 0.9)

train_df = full_df.iloc[:train_end].reset_index(drop=True)
val_df = full_df.iloc[train_end:val_end].reset_index(drop=True)
test_df = full_df.iloc[val_end:].reset_index(drop=True)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# --- CELL 5: Optuna Optimizasyonu ---
import numpy as np
import optuna
import config as cfg
from env import OFITradingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# -- Ayarlar --
N_TRIALS = 30
TIMESTEPS_PER_TRIAL = 50_000

def objective(trial):
    # Hiperparametre uzayi
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    n_epochs = trial.suggest_int("n_epochs", 3, 15)

    # batch_size n_steps'i tam bolmeli
    if n_steps % batch_size != 0:
        for bs in [512, 256, 128, 64, 32]:
            if n_steps % bs == 0:
                batch_size = bs
                break

    # Ortam
    train_env = DummyVecEnv([lambda: Monitor(OFITradingEnv(df=train_df))])
    val_env = DummyVecEnv([lambda: Monitor(OFITradingEnv(df=val_df))])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=cfg.CLIP_OBS)
    val_env = VecNormalize(val_env, norm_obs=True, norm_reward=False, clip_obs=cfg.CLIP_OBS)

    try:
        model = PPO(
            "MlpPolicy", train_env,
            learning_rate=lr, n_steps=n_steps, batch_size=batch_size,
            n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=clip_range, ent_coef=ent_coef,
            verbose=0, device="auto"
        )
        model.learn(total_timesteps=TIMESTEPS_PER_TRIAL)

        # Validation
        val_env.obs_rms = train_env.obs_rms
        val_env.ret_rms = train_env.ret_rms
        val_env.training = False
        val_env.norm_reward = False

        mean_reward, std_reward = evaluate_policy(model, val_env, n_eval_episodes=5, deterministic=True)

    except Exception as e:
        print(f"Trial {trial.number} basarisiz: {e}")
        return float("-inf")
    finally:
        train_env.close()
        val_env.close()

    print(f"Trial {trial.number}: mean_reward={mean_reward:.5f} (+/- {std_reward:.5f})")
    return mean_reward

# Calistir
study = optuna.create_study(direction="maximize", study_name="PPO_OFI_HFT")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# --- CELL 6: Sonuclar ---
print("\n" + "="*60)
print("OPTIMIZASYON TAMAMLANDI")
print("="*60)
print(f"\nEn Iyi Trial    : #{study.best_trial.number}")
print(f"En Iyi Reward   : {study.best_value:.6f}")
print(f"\nEn Iyi Hiperparametreler:")
print("-"*40)
for key, value in study.best_params.items():
    if isinstance(value, float):
        print(f"  {key:<20}: {value:.6f}")
    else:
        print(f"  {key:<20}: {value}")

print("\n" + "="*60)
print("config.py ICIN KOPYALA-YAPISTIR:")
print("="*60)
param_map = {
    "learning_rate": "LEARNING_RATE",
    "n_steps": "N_STEPS",
    "batch_size": "BATCH_SIZE",
    "n_epochs": "N_EPOCHS",
    "gamma": "GAMMA",
    "gae_lambda": "GAE_LAMBDA",
    "clip_range": "CLIP_RANGE",
    "ent_coef": "ENT_COEF",
}
for optuna_key, config_key in param_map.items():
    if optuna_key in study.best_params:
        val = study.best_params[optuna_key]
        print(f"{config_key} = {val}")

# Sonuclari CSV olarak indir
results_df = study.trials_dataframe()
results_df.to_csv("optuna_results.csv", index=False)
files.download("optuna_results.csv")
print("\noptuna_results.csv indirildi!")
