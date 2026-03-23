"""
optimize_agent.py -- Optuna Hyperparameter Optimization for PPO Agent

Uses Optuna to automatically search the best PPO hyperparameters
by training short trials and evaluating on a held-out validation set.

Usage:
    pip install optuna
    python optimize_agent.py                    # Default: 30 trials
    python optimize_agent.py --trials 50        # Custom trial count
    python optimize_agent.py --timesteps 100000 # Custom timesteps per trial
"""

import os
import argparse
import numpy as np
import pandas as pd

try:
    import optuna
except ImportError:
    raise ImportError("Optuna not installed. Run: pip install optuna")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

import config as cfg
from env import OFITradingEnv
from logger import get_logger

logger = get_logger("Optimizer")


def load_and_split_data():
    """Loads real market data and splits into train/validation/test."""
    if not os.path.exists(cfg.DATA_PATH):
        raise FileNotFoundError(
            f"Real market data not found at {cfg.DATA_PATH}. "
            "Run data_collector.py first."
        )
    
    full_df = pd.read_csv(cfg.DATA_PATH)
    n = len(full_df)
    
    # 70% train, 10% validation (for Optuna), 20% test (untouched)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)
    
    train_df = full_df.iloc[:train_end].reset_index(drop=True)
    val_df = full_df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = full_df.iloc[val_end:].reset_index(drop=True)
    
    logger.info(f"Data loaded: {n:,} rows -> "
                 f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    
    return train_df, val_df, test_df


def create_objective(train_df, val_df, timesteps_per_trial):
    """Creates an Optuna objective function closure with the given data."""
    
    def objective(trial: optuna.Trial) -> float:
        """
        Single Optuna trial: samples hyperparameters, trains PPO,
        evaluates on validation set, returns mean reward.
        """
        # ----------------------------------------------------------
        # HYPERPARAMETER SEARCH SPACE
        # ----------------------------------------------------------
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
        clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
        gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
        n_epochs = trial.suggest_int("n_epochs", 3, 15)
        
        # batch_size must divide n_steps evenly
        if n_steps % batch_size != 0:
            # Adjust batch_size down to nearest valid divisor
            for bs in [512, 256, 128, 64, 32]:
                if n_steps % bs == 0 and bs <= n_steps:
                    batch_size = bs
                    break
        
        logger.info(
            f"Trial {trial.number}: lr={learning_rate:.6f}, n_steps={n_steps}, "
            f"batch={batch_size}, gamma={gamma:.4f}, ent={ent_coef:.5f}, "
            f"clip={clip_range}, gae={gae_lambda:.3f}, epochs={n_epochs}"
        )
        
        # ----------------------------------------------------------
        # ENVIRONMENT SETUP
        # ----------------------------------------------------------
        def make_train_env():
            return Monitor(OFITradingEnv(df=train_df))
        
        def make_val_env():
            return Monitor(OFITradingEnv(df=val_df))
        
        train_env = DummyVecEnv([make_train_env])
        val_env = DummyVecEnv([make_val_env])
        
        train_env = VecNormalize(
            train_env, norm_obs=True, norm_reward=True, clip_obs=cfg.CLIP_OBS
        )
        val_env = VecNormalize(
            val_env, norm_obs=True, norm_reward=False, clip_obs=cfg.CLIP_OBS
        )
        
        # ----------------------------------------------------------
        # TRAINING
        # ----------------------------------------------------------
        try:
            model = PPO(
                "MlpPolicy",
                train_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=0,
            )
            
            model.learn(total_timesteps=timesteps_per_trial)
            
        except Exception as e:
            logger.error(f"Trial {trial.number} training failed: {e}")
            return float("-inf")
        
        # ----------------------------------------------------------
        # EVALUATION on validation set
        # ----------------------------------------------------------
        try:
            # Sync normalization stats from training env to validation env
            val_env.obs_rms = train_env.obs_rms
            val_env.ret_rms = train_env.ret_rms
            val_env.training = False
            val_env.norm_reward = False
            
            mean_reward, std_reward = evaluate_policy(
                model, val_env, n_eval_episodes=5, deterministic=True
            )
            
        except Exception as e:
            logger.error(f"Trial {trial.number} evaluation failed: {e}")
            return float("-inf")
        finally:
            train_env.close()
            val_env.close()
        
        logger.info(
            f"Trial {trial.number} result: mean_reward={mean_reward:.5f} "
            f"(+/- {std_reward:.5f})"
        )
        
        return mean_reward
    
    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna PPO Hyperparameter Optimization")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Training timesteps per trial (default: 50,000)")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Optuna PPO Hyperparameter Optimization")
    logger.info(f"Trials     : {args.trials}")
    logger.info(f"Timesteps  : {args.timesteps:,} per trial")
    logger.info("=" * 60)
    
    # Load data
    train_df, val_df, test_df = load_and_split_data()
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="PPO_OFI_HFT",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    
    objective = create_objective(train_df, val_df, args.timesteps)
    
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    
    # ----------------------------------------------------------
    # RESULTS
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nBest Trial     : #{study.best_trial.number}")
    print(f"Best Reward    : {study.best_value:.6f}")
    print(f"\nBest Hyperparameters:")
    print("-" * 40)
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key:<20}: {value:.6f}")
        else:
            print(f"  {key:<20}: {value}")
    print("=" * 60)
    
    # Save best params to config suggestion
    print("\nconfig.py icin onerilen guncellemeler:")
    print("-" * 40)
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
            if isinstance(val, float):
                print(f"  {config_key} = {val}")
            else:
                print(f"  {config_key} = {val}")
    
    # Save study results to CSV
    results_path = os.path.join(cfg.LOGS_DIR, "optuna_results.csv")
    os.makedirs(cfg.LOGS_DIR, exist_ok=True)
    study.trials_dataframe().to_csv(results_path, index=False)
    logger.info(f"All trial results saved to: {results_path}")


if __name__ == "__main__":
    main()
