import numpy as np
from env import OFITradingEnv
from logger import get_logger

logger = get_logger("EnvTest")

def test_environment():
    """
    Test the Custom Trading Environment with dummy data to verify state and reward transitions.
    """
    # Initialize the environment
    env = OFITradingEnv(commission_rate=0.0004, render_mode="ansi")
    
    # Reset Environment
    observation, info = env.reset()
    logger.info(f"Initial Observation: {observation}")
    
    # Simulate Market Tick 1 (Price goes up slightly, Spread tight)
    # OFI is highly positive (+5.0) indicating BUY pressure
    env.update_market_data(ofi=5.0, bid=71500.0, ask=71500.1)
    
    # Action 1: BUY (We see strong OFI, agent decides to enter Long)
    observation, reward, terminated, truncated, info = env.step(action=1)
    env.render()
    logger.info(f"Action: Buy | Reward: {reward:.5f} | Info: {info}")
    
    # Simulate Market Tick 2 (Price goes up significantly)
    # OFI stays positive (+2.0)
    env.update_market_data(ofi=2.0, bid=71600.0, ask=71600.2)
    
    # Action 0: HOLD (Ride the profit)
    observation, reward, terminated, truncated, info = env.step(action=0)
    env.render()
    logger.info(f"Action: Hold | Reward: {reward:.5f} | Info: {info}")
    
    # Simulate Market Tick 3 (Price goes up, but OFI turns very negative)
    # OFI is heavily negative (-8.0) indicating incoming SELL pressure
    env.update_market_data(ofi=-8.0, bid=71620.0, ask=71620.5)
    
    # Action 2: SELL (Agent detects negative OFI and closes the Long position, essentially going Flat/Shortting later if needed)
    # Since we are Long, action 2 will Close Long and open Short. Let's see the reward.
    observation, reward, terminated, truncated, info = env.step(action=2)
    env.render()
    logger.info(f"Action: Sell (Close Long) | Reward: {reward:.5f} | Info: {info}")
    
    logger.info("Environment sanity check completed successfully.")

if __name__ == "__main__":
    test_environment()
