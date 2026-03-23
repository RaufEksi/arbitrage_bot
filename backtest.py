import os
import numpy as np
import pandas as pd
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
    
    mid_price = 70000.0
    spread = 0.5
    
    for _ in range(steps):
        ofi = np.random.normal(loc=0.0, scale=3.0) 
        price_change = (ofi * 0.1) + np.random.normal(loc=0, scale=0.5)
        mid_price += price_change
        spread = max(0.1, np.random.normal(loc=0.5, scale=0.2))
        
        bid = round(mid_price - (spread / 2.0), 2)
        ask = round(mid_price + (spread / 2.0), 2)
        
        data.append((ofi, bid, ask))
        
    return data

def run_backtest(model_path, data, vec_normalize_path=None):
    """
    Simulates trading over a given dataset and returns the history.
    Uses VecNormalize stats if available for proper observation scaling.
    
    'data' can be:
      - list of tuples (ofi, bid, ask)  -- synthetic mode
      - pd.DataFrame with required columns -- real data mode
    """
    logger.info(f"Loading trained model from: {model_path}")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}. Ensure you have trained it first! Error: {e}")
        return None, None, None
    
    # Determine data mode
    use_dataframe = isinstance(data, pd.DataFrame)
    
    if use_dataframe:
        raw_env = OFITradingEnv(commission_rate=0.0004, df=data)
        total_steps = len(data) - 1
    else:
        raw_env = OFITradingEnv(commission_rate=0.0004, max_steps=0)
        total_steps = len(data)
    
    vec_env = DummyVecEnv([lambda: raw_env])
    
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        logger.info(f"VecNormalize stats loaded from: {vec_normalize_path} (training=False)")
    else:
        logger.warning("No VecNormalize stats found. Running without observation normalization.")
    
    obs = vec_env.reset()
    
    actions_taken = []
    rewards_history = []
    cumulative_pnl = [0.0]
    cumulative_raw_reward = 0.0
    
    logger.info(f"Starting deterministic simulation for {total_steps} steps...")
    
    for step_idx in range(total_steps):
        # In synthetic mode, push data manually
        if not use_dataframe:
            ofi, bid, ask = data[step_idx]
            raw_env.update_market_data(ofi=ofi, bid=bid, ask=ask)
        
        # Get normalized observation
        obs = vec_env.normalize_obs(np.array([raw_env.state]))
        
        action, _states = model.predict(obs, deterministic=True)
        action_val = int(action[0])
        actions_taken.append(action_val)
        
        # In DataFrame mode, step() calls _load_tick_from_df() internally
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

def plot_equity_curve(cumulative_pnl, data_source=""):
    """
    Plots and saves the equity curve using Matplotlib.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_pnl, label="Cumulative PnL (Rewards)", color="#1f77b4", linewidth=2)
        plt.fill_between(range(len(cumulative_pnl)), cumulative_pnl, color="#1f77b4", alpha=0.1)
        
        title = f"HFT Agent Equity Curve ({data_source})"
        plt.title(title, fontsize=14, pad=15)
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
    MODELS_DIR = "./models/"
    DATA_DIR = "./data/"
    MODEL_PATH = os.path.join(MODELS_DIR, "best_model.zip")
    VEC_NORMALIZE_PATH = os.path.join(MODELS_DIR, "vec_normalize.pkl")
    REAL_DATA_PATH = os.path.join(DATA_DIR, "btcusdt_ofi_data.csv")
    
    if not os.path.exists(MODEL_PATH):
        logger.warning("best_model.zip not found. Falling back to final_trading_model.zip")
        MODEL_PATH = os.path.join(MODELS_DIR, "final_trading_model.zip")
    
    # ------------------------------------------------------------------
    # DATA SOURCE: Real CSV if available, otherwise synthetic
    # ------------------------------------------------------------------
    if os.path.exists(REAL_DATA_PATH):
        logger.info(f"Real market data found: {REAL_DATA_PATH}")
        full_df = pd.read_csv(REAL_DATA_PATH)
        split_idx = int(len(full_df) * 0.8)
        test_data = full_df.iloc[split_idx:].reset_index(drop=True)
        logger.info(f"Out-of-sample test: last {len(test_data):,} rows "
                     f"(of {len(full_df):,} total, 20% split).")
        data_source = "Real Binance Data (Out-of-Sample)"
    else:
        logger.info("No real data found. Generating synthetic backtest data...")
        STEPS = 5000
        test_data = generate_test_data(steps=STEPS)
        data_source = "Synthetic Data"
    
    # Run backtest
    actions, cum_pnl, step_rewards = run_backtest(MODEL_PATH, test_data, VEC_NORMALIZE_PATH)
    
    if actions and cum_pnl:
        action_counts = Counter(actions)
        total_profit = cum_pnl[-1]
        
        ANNUALIZATION_FACTOR = 252 * 24 * 60
        sharpe_ratio = calculate_sharpe(step_rewards, ANNUALIZATION_FACTOR)
        
        logger.info(f"\n========== BACKTEST RESULTS ({data_source}) ==========")
        logger.info(f"Total Steps        : {len(actions)}")
        logger.info(f"Net Profit/Reward  : {total_profit:>.5f}")
        logger.info(f"Ann. Sharpe Ratio  : {sharpe_ratio:>.3f}")
        logger.info("-" * 50)
        logger.info("Action Distribution:")
        logger.info(f"  Hold (0) : {action_counts.get(0, 0)}")
        logger.info(f"  Buy  (1) : {action_counts.get(1, 0)}")
        logger.info(f"  Sell (2) : {action_counts.get(2, 0)}")
        logger.info("=" * 50)
        
        plot_equity_curve(cum_pnl, data_source)

