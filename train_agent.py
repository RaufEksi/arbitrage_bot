import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

import config as cfg
from env import OFITradingEnv
from logger import get_logger

logger = get_logger("TrainAgent")

os.makedirs(cfg.MODELS_DIR, exist_ok=True)
os.makedirs(cfg.LOGS_DIR, exist_ok=True)


def train():
    logger.info("V4 Training Pipeline with Train/Test Split")
    
    # ------------------------------------------------------------------
    # DATA LOADING & TRAIN/TEST SPLIT
    # ------------------------------------------------------------------
    if os.path.exists(cfg.DATA_PATH):
        full_df = pd.read_csv(cfg.DATA_PATH)
        split_idx = int(len(full_df) * cfg.TRAIN_TEST_SPLIT)
        train_df = full_df.iloc[:split_idx].reset_index(drop=True)
        test_df = full_df.iloc[split_idx:].reset_index(drop=True)
        logger.info(f"Real data loaded: {len(full_df):,} rows -> "
                     f"Train: {len(train_df):,} | Test: {len(test_df):,}")
        
        def make_train_env():
            return OFITradingEnv(df=train_df)
        def make_eval_env():
            return OFITradingEnv(df=test_df)
    else:
        logger.warning(f"Real data not found at {cfg.DATA_PATH}. Using synthetic mode.")
        def make_train_env():
            return OFITradingEnv()
        def make_eval_env():
            return OFITradingEnv()
    
    # ------------------------------------------------------------------
    # ENVIRONMENT WRAPPING: DummyVecEnv -> VecNormalize
    # ------------------------------------------------------------------
    env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=cfg.CLIP_OBS)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=cfg.CLIP_OBS)
    logger.info(f"VecNormalize active: clip_obs={cfg.CLIP_OBS}")
    
    # ------------------------------------------------------------------
    # CALLBACK: Save the best model every 10,000 steps based on rewards
    # ------------------------------------------------------------------
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=cfg.MODELS_DIR,
        log_path=cfg.LOGS_DIR, 
        eval_freq=cfg.EVAL_FREQ,
        deterministic=True, 
        render=False
    )
    
    logger.info("Setting up PPO Agent (V4 Hyperparameters from config.py)...")
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
        tensorboard_log=cfg.LOGS_DIR
    )
    
    TOTAL_TIMESTEPS = cfg.TOTAL_TIMESTEPS_LOCAL
    
    logger.info(f"Starting Training for {TOTAL_TIMESTEPS} timesteps...")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="PPO_OFI_Trading"
    )
    
    logger.info("Training Finished.")
    
    # Save the final model explicitly
    final_model_path = os.path.join(cfg.MODELS_DIR, "final_trading_model")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}.zip")
    
    vec_norm_path = os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl")
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
    raw_env = OFITradingEnv(render_mode="ansi")
    vec_env = DummyVecEnv([lambda: raw_env])
    
    vec_norm_path = os.path.join(cfg.MODELS_DIR, "vec_normalize.pkl")
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
    best_model_path = os.path.join(cfg.MODELS_DIR, "best_model.zip")
    if os.path.exists(best_model_path):
        evaluate_model(model_path=best_model_path, steps=10)
    else:
        evaluate_model(model_path=os.path.join(cfg.MODELS_DIR, "final_trading_model.zip"), steps=10)
