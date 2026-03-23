import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from env import OFITradingEnv
from logger import get_logger

logger = get_logger("TrainAgent")

# Directory setup for Saving Models and TensorBoard Logs
MODELS_DIR = "./models/"
LOGS_DIR = "./logs/"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def make_env():
    """
    Creates and returns an instance of the OFITradingEnv.
    Used by Stable-Baselines3's DummyVecEnv.
    """
    return OFITradingEnv(commission_rate=0.0004)


def train():
    logger.info("Initializing Stable-Baselines3 Environment...")
    
    # ------------------------------------------------------------------
    # ENVIRONMENT WRAPPING: DummyVecEnv -> VecNormalize
    # ------------------------------------------------------------------
    # 1. Wrap in DummyVecEnv (required by SB3)
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    # 2. VecNormalize: Normalizes observations and rewards on-the-fly
    #    - norm_obs=True:  Normalizes OFI (scale ~0-10) and Spread (scale ~0.01)
    #                      to the same magnitude, preventing network blindness
    #    - norm_reward=True: Stabilizes reward signal for PPO gradient updates
    #    - clip_obs=10.0:  Clips extreme outlier observations (flash crashes etc.)
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
    
    logger.info("Setting up PPO Agent (V2 Hyperparameters)...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,        # V2: Slower, more stable convergence
        n_steps=4096,              # V2: Longer rollouts = more stable gradients
        batch_size=128,            # V2: Larger batch for 12-dim state
        n_epochs=10,
        gamma=0.95,                # V2: HFT doesn't need distant future (was 0.99)
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,             # V2: More exploration to prevent premature convergence
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
    Loads a trained model and evaluates it using dummy environment data
    """
    logger.info("====================================")
    logger.info(f"Evaluating Model: {model_path}")
    logger.info("====================================")
    
    # Load model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return
        
    # Standard non-vectorized env to push dummy market data manually
    test_env = OFITradingEnv(commission_rate=0.0004, render_mode="ansi")
    obs, info = test_env.reset()
    
    # Simulating 10 ticks of market data manually for the evaluation
    # Format: (OFI, Bid, Ask)
    dummy_market_data = [
        ( 1.5, 70000.0, 70000.1),  # Mild Buy
        ( 5.0, 70000.5, 70000.6),  # Strong Buy
        ( 6.0, 70001.0, 70001.2),  # Very Strong Buy
        (-0.5, 70002.0, 70002.1),  # Flattening
        (-3.0, 70000.0, 70000.1),  # Strong Sell
        (-5.0, 69998.0, 69998.1),  # Very Strong Sell
        ( 0.0, 69995.0, 69995.1),  # Flat
        ( 4.0, 69996.0, 69996.1),  # Buy Reversal
        ( 2.0, 69998.0, 69998.1),  # Mild Buy
        (-1.0, 69999.0, 69999.1),  # Flat
    ]
    
    for i in range(min(steps, len(dummy_market_data))):
        ofi, bid, ask = dummy_market_data[i]
        
        # 1. Update Market Env implicitly first (like a websocket would)
        test_env.update_market_data(ofi=ofi, bid=bid, ask=ask)
        
        # We must re-fetch the raw state to feed into the model since market moved
        # Stable Baselines expects a batched environment input, so we wrap it
        obs = np.array([test_env.state])
        
        # 2. Agent decides action deterministically
        action, _states = model.predict(obs, deterministic=True)
        # Unwrap the batched action prediction
        action_val = int(action[0])
        
        # 3. Environment Step
        obs, reward, terminated, truncated, info = test_env.step(action_val)
        
        # Log
        action_name = ["HOLD", "BUY", "SELL"][action_val]
        logger.info(f"Tick {i+1} | OFI: {ofi:>4.1f} | Action Given: {action_name:<4} | R: {reward:>.5f} | PnL: {info['pnl']:>.5f}")
        
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
