import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from env import OFITradingEnv
from logger import get_logger

logger = get_logger("TrainAgent")

# Configuration
MODELS_DIR = "./models/"
LOGS_DIR = "./logs/"
DATA_PATH = "./data/btcusdt_ofi_data.csv"

# Directory setup for Saving Models and TensorBoard Logs
MODELS_DIR = "./models/"
LOGS_DIR = "./logs/"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def train():
    logger.info("V4 Training Pipeline with Train/Test Split")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # ------------------------------------------------------------------
    # DATA LOADING & TRAIN/TEST SPLIT (80/20)
    # ------------------------------------------------------------------
    if os.path.exists(DATA_PATH):
        full_df = pd.read_csv(DATA_PATH)
        split_idx = int(len(full_df) * 0.8)
        train_df = full_df.iloc[:split_idx].reset_index(drop=True)
        test_df = full_df.iloc[split_idx:].reset_index(drop=True)
        logger.info(f"Real data loaded: {len(full_df):,} rows -> "
                     f"Train: {len(train_df):,} | Test: {len(test_df):,}")
        
        def make_train_env():
            return OFITradingEnv(commission_rate=0.0004, df=train_df)
        def make_eval_env():
            return OFITradingEnv(commission_rate=0.0004, df=test_df)
    else:
        logger.warning(f"Real data not found at {DATA_PATH}. Using synthetic mode.")
        def make_train_env():
            return OFITradingEnv(commission_rate=0.0004)
        def make_eval_env():
            return OFITradingEnv(commission_rate=0.0004)
    
    # ------------------------------------------------------------------
    # ENVIRONMENT WRAPPING: DummyVecEnv -> VecNormalize
    # ------------------------------------------------------------------
    env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    logger.info("VecNormalize active: norm_obs=True, norm_reward=True, clip_obs=10.0")
    
    # ------------------------------------------------------------------
    # CALLBACK: Save the best model every 10,000 steps based on rewards
    # ------------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=MODELS_DIR,
        log_path=LOGS_DIR, 
        eval_freq=10000,           # Evaluate every 10k steps
        deterministic=True, 
        render=False
    )
    
    # ------------------------------------------------------------------
    # PPO HYPERPARAMETERS FOR FINANCIAL TIME SERIES
    # ------------------------------------------------------------------
    # Financial data is notoriously noisy. To prevent catastrophic forgetting 
    # and premature convergence to local optima (like always holding), 
    # we tune PPO's hyperparameters:
    
    logger.info("Setting up PPO Agent (V4 Hyperparameters)...")
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
        ent_coef=0.05,             # V4: Higher exploration (was 0.02)
        verbose=1,
        tensorboard_log=LOGS_DIR
    )
    
    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    TOTAL_TIMESTEPS = 20_000 # 100k for demonstration. Real-world needs 1M+
    
    logger.info(f"Starting Training for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="PPO_OFI_Trading"
    )
    
    logger.info("Training Finished.")
    
    # Save the final model explicitly
    final_model_path = os.path.join(MODELS_DIR, "final_trading_model")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}.zip")
    
    # ------------------------------------------------------------------
    # CRITICAL: Save VecNormalize statistics alongside the model
    # Without these stats, the model will receive un-normalized inputs
    # at inference time and produce garbage predictions.
    # ------------------------------------------------------------------
    vec_norm_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")
    env.save(vec_norm_path)
    logger.info(f"VecNormalize stats saved to: {vec_norm_path}")
    
    return model

def evaluate_model(model_path: str, steps: int = 10):
    """
    Loads a trained model and evaluates it with VecNormalize-scaled observations.
    """
    logger.info("====================================")
    logger.info(f"Evaluating Model: {model_path}")
    logger.info("====================================")
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return
    
    # Create raw env and wrap with VecNormalize (inference mode)
    raw_env = OFITradingEnv(commission_rate=0.0004, render_mode="ansi")
    vec_env = DummyVecEnv([lambda: raw_env])
    
    vec_norm_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")
    if os.path.exists(vec_norm_path):
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False      # Freeze normalization stats
        vec_env.norm_reward = False    # Show raw rewards
        logger.info(f"VecNormalize stats loaded for evaluation (training=False)")
    else:
        logger.warning("No VecNormalize stats found! Agent will see raw observations.")
    
    obs = vec_env.reset()
    
    dummy_market_data = [
        ( 1.5, 70000.0, 70000.1),
        ( 5.0, 70000.5, 70000.6),
        ( 6.0, 70001.0, 70001.2),
        (-0.5, 70002.0, 70002.1),
        (-3.0, 70000.0, 70000.1),
        (-5.0, 69998.0, 69998.1),
        ( 0.0, 69995.0, 69995.1),
        ( 4.0, 69996.0, 69996.1),
        ( 2.0, 69998.0, 69998.1),
        (-1.0, 69999.0, 69999.1),
    ]
    
    for i in range(min(steps, len(dummy_market_data))):
        ofi, bid, ask = dummy_market_data[i]
        
        # 1. Update market data on the raw (inner) environment
        raw_env.update_market_data(ofi=ofi, bid=bid, ask=ask)
        
        # 2. Get normalized observation through VecNormalize
        obs = vec_env.normalize_obs(np.array([raw_env.state]))
        
        # 3. Agent predicts action using normalized observation
        action, _states = model.predict(obs, deterministic=True)
        action_val = int(action[0])
        
        # 4. Step the raw env
        raw_obs, reward, terminated, truncated, info = raw_env.step(action_val)
        
        action_name = ["HOLD", "BUY", "SELL"][action_val]
        logger.info(f"Tick {i+1} | OFI: {ofi:>4.1f} | Action: {action_name:<4} | "
                     f"R: {reward:>.5f} | PnL: {info['pnl']:>.5f}")
        
    logger.info("Evaluation Complete.")

if __name__ == "__main__":
    # 1. Train the agent
    train()
    
    # 2. Evaluate the best model created by EvalCallback
    best_model_path = os.path.join(MODELS_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        evaluate_model(model_path=best_model_path, steps=10)
    else:
        # Fallback to final model
        evaluate_model(model_path=os.path.join(MODELS_DIR, "final_trading_model.zip"), steps=10)
