import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import OFITradingEnv
from logger import get_logger

logger = get_logger("Backtester")

def generate_test_data(steps=1000):
    """
    Generates a synthetic price and OFI dataset representing HFT order book dynamics.
    Returns a list of tuples: (ofi, bid, ask)
    """
    np.random.seed(42)  # For deterministic behavior
    data = []
    
    # Starting values for BTC-like simulation
    mid_price = 70000.0
    spread = 0.5
    
    for _ in range(steps):
        # Simulate OFI as a noisy mean-reverting signal
        ofi = np.random.normal(loc=0.0, scale=3.0) 
        
        # Price tends to move in the direction of OFI (market impact)
        price_change = (ofi * 0.1) + np.random.normal(loc=0, scale=0.5)
        mid_price += price_change
        
        # Introduce some spread variance
        spread = max(0.1, np.random.normal(loc=0.5, scale=0.2))
        
        bid = round(mid_price - (spread / 2.0), 2)
        ask = round(mid_price + (spread / 2.0), 2)
        
        data.append((ofi, bid, ask))
        
    return data

def run_backtest(model_path, data, vec_normalize_path=None):
    """
    Simulates trading over a given dataset and returns the history.
    Uses VecNormalize stats if available for proper observation scaling.
    """
    logger.info(f"Loading trained model from: {model_path}")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Ensure you have trained it first! Error: {e}")
        return None, None, None
    
    # ------------------------------------------------------------------
    # ENVIRONMENT SETUP WITH VECNORMALIZE (Inference Mode)
    # ------------------------------------------------------------------
    # We need a raw OFITradingEnv to push market data manually,
    # BUT we also need VecNormalize to scale observations for the model.
    raw_env = OFITradingEnv(commission_rate=0.0004, max_steps=0)  # max_steps=0 = no truncation
    vec_env = DummyVecEnv([lambda: raw_env])
    
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        # CRITICAL: Freeze normalization stats during testing
        vec_env.training = False
        # Show raw (un-normalized) rewards for realistic PnL reporting
        vec_env.norm_reward = False
        logger.info(f"VecNormalize stats loaded from: {vec_normalize_path} (training=False)")
    else:
        logger.warning("No VecNormalize stats found. Running without observation normalization.")
    
    obs = vec_env.reset()
    
    # Tracking metrics
    actions_taken = []
    rewards_history = []
    cumulative_pnl = [0.0]
    cumulative_raw_reward = 0.0
    
    logger.info(f"Starting deterministic simulation for {len(data)} steps...")
    
    for step_idx, (ofi, bid, ask) in enumerate(data):
        # 1. Update Market Context on the RAW (inner) environment
        raw_env.update_market_data(ofi=ofi, bid=bid, ask=ask)
        
        # 2. Get the normalized observation through VecNormalize
        #    VecNormalize will scale the raw state using training stats
        obs = vec_env.normalize_obs(np.array([raw_env.state]))
        
        # 3. Agent predicts action deterministically
        action, _states = model.predict(obs, deterministic=True)
        action_val = int(action[0])
        actions_taken.append(action_val)
        
        # 4. Execute action on the RAW env to get true (un-normalized) reward
        raw_obs, reward, terminated, truncated, info = raw_env.step(action_val)
        
        rewards_history.append(reward)
        cumulative_raw_reward += reward
        cumulative_pnl.append(cumulative_raw_reward)
        
        if terminated or truncated:
            logger.warning(f"Backtest episode ended early at step {step_idx}.")
            break
            
    return actions_taken, cumulative_pnl, rewards_history

def calculate_sharpe(rewards, annualization_factor):
    """
    Calculates Annualized Sharpe Ratio.
    Assumes Risk-Free Rate is 0 for simplicity.
    """
    returns = np.array(rewards)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.0
        
    sharpe = (mean_return / std_return) * np.sqrt(annualization_factor)
    return sharpe

def plot_equity_curve(cumulative_pnl):
    """
    Plots and saves the equity curve using Matplotlib.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_pnl, label="Cumulative PnL (Rewards)", color="#1f77b4", linewidth=2)
        
        # Fill area under curve
        plt.fill_between(range(len(cumulative_pnl)), cumulative_pnl, color="#1f77b4", alpha=0.1)
        
        plt.title("HFT Agent Equity Curve (Backtest on Test Dataset)", fontsize=14, pad=15)
        plt.xlabel("Timesteps (Ticks)", fontsize=12)
        plt.ylabel("Cumulative Net Return", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="upper left")
        plt.tight_layout()
        
        save_path = "equity_curve.png"
        plt.savefig(save_path, dpi=300)
        logger.info(f"Equity curve plot successfully saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to plot equity curve: {e}")

if __name__ == "__main__":
    # Backtest Configuration
    MODELS_DIR = "./models/"
    MODEL_PATH = os.path.join(MODELS_DIR, "best_model.zip")
    VEC_NORMALIZE_PATH = os.path.join(MODELS_DIR, "vec_normalize.pkl")
    
    # Fallback to final model if best model is missing
    if not os.path.exists(MODEL_PATH):
        logger.warning("best_model.zip not found. Falling back to final_trading_model.zip")
        MODEL_PATH = os.path.join(MODELS_DIR, "final_trading_model.zip")
        
    STEPS = 5000
    
    # 1. Generate Fake HFT Data 
    logger.info("Generating continuous synthetic backtest data...")
    test_data = generate_test_data(steps=STEPS)
    
    # 2. Run Backtest simulation (with VecNormalize stats)
    actions, cum_pnl, step_rewards = run_backtest(MODEL_PATH, test_data, VEC_NORMALIZE_PATH)
    
    if actions and cum_pnl:
        # 3. Calculate metrics
        action_counts = Counter(actions)
        total_profit = cum_pnl[-1]
        
        # Annualization factor requested: sqrt(252 * 24 * 60)
        # We pass 252 * 24 * 60 inside the sqrt calculation
        ANNUALIZATION_FACTOR = 252 * 24 * 60
        sharpe_ratio = calculate_sharpe(step_rewards, ANNUALIZATION_FACTOR)
        
        # 4. Display terminal report
        logger.info("\n========== BACKTEST RESULTS ==========")
        logger.info(f"Total Steps        : {len(actions)}")
        logger.info(f"Net Profit/Reward  : {total_profit:>.5f}")
        logger.info(f"Ann. Sharpe Ratio  : {sharpe_ratio:>.3f}")
        logger.info("-" * 38)
        logger.info("Action Distribution:")
        logger.info(f"  Hold (0) : {action_counts.get(0, 0)}")
        logger.info(f"  Buy  (1) : {action_counts.get(1, 0)}")
        logger.info(f"  Sell (2) : {action_counts.get(2, 0)}")
        logger.info("======================================")
        
        # 5. Visualization
        plot_equity_curve(cum_pnl)
