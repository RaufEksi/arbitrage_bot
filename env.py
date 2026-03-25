import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, Any, Tuple
from logger import get_logger
import config as cfg

logger = get_logger("OFITradingEnv")


class OFITradingEnv(gym.Env):
    """
    V4 HFT Trading Environment — Performance Optimized + Feature Engineering.
    
    Observation Space (14-dim Continuous):
        [0-4]:  OFI lookback window (last 5 ticks)
        [5]:    OFI Exponential Moving Average (EMA-20)
        [6]:    Current Spread
        [7]:    Spread change rate (ΔSpread)
        [8]:    Current Position (-1, 0, 1)
        [9]:    Unrealized PnL
        [10]:   Holding time (normalized by max_steps)
        [11]:   Step progress (current_step / max_steps)
        [12]:   OBI — Order Book Imbalance (bid_qty-ask_qty)/(bid_qty+ask_qty)
        [13]:   Volatility — Rolling std of mid_price
        
    Action Space (Discrete):
        0: Hold (Do nothing)
        1: Buy (Go Long / Close Short)
        2: Sell (Go Short / Close Long)
        
    Reward Mechanism (V4 — Balanced):
        - Differential ΔPnL for holding positions
        - Commission + slippage cost on every trade
        - Realized PnL on position close
        - Soft overtrading penalty: gentle discouragement of excessive trading
        - Soft redundant action penalty: gentle discouragement of no-op actions
    """
    
    metadata = {"render_modes": ["ansi"]}
    
    # --- Constants from config.py ---
    OVERTRADE_WINDOW = cfg.OVERTRADE_WINDOW
    OVERTRADE_MAX = cfg.OVERTRADE_MAX
    OVERTRADE_PENALTY = cfg.OVERTRADE_PENALTY
    REDUNDANT_PENALTY = cfg.REDUNDANT_PENALTY
    OFI_LOOKBACK = cfg.OFI_LOOKBACK
    EMA_SPAN = cfg.EMA_SPAN
    
    def __init__(self, commission_rate: float = cfg.COMMISSION_RATE, render_mode: Optional[str] = None,
                 max_steps: int = cfg.DEFAULT_MAX_STEPS, df: Optional[pd.DataFrame] = None):
        super(OFITradingEnv, self).__init__()
        
        self.commission_rate = commission_rate
        self.render_mode = render_mode
        
        # --- DataFrame Mode: pre-cache as numpy for O(1) access ---
        self._df_ofi = None
        self._df_bid = None
        self._df_ask = None
        self._df_len = 0
        
        if df is not None:
            required_cols = {"bid_price", "ask_price", "ofi", "spread"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"DataFrame missing required columns: {missing}")
            self._df_ofi = df["ofi"].to_numpy(dtype=np.float64)
            self._df_bid = df["bid_price"].to_numpy(dtype=np.float64)
            self._df_ask = df["ask_price"].to_numpy(dtype=np.float64)
            
            # --- Feature Engineering: OBI and Volatility ---
            if "bid_qty" in df.columns and "ask_qty" in df.columns:
                bid_qty = df["bid_qty"].to_numpy(dtype=np.float64)
                ask_qty = df["ask_qty"].to_numpy(dtype=np.float64)
                total_qty = bid_qty + ask_qty
                self._df_obi = np.where(total_qty > 0, (bid_qty - ask_qty) / total_qty, 0.0)
            else:
                self._df_obi = np.zeros(len(df), dtype=np.float64)
                logger.warning("bid_qty/ask_qty not found — OBI set to 0.")
            
            mid_price = (df["bid_price"].to_numpy(dtype=np.float64) + 
                         df["ask_price"].to_numpy(dtype=np.float64)) / 2.0
            vol_window = getattr(cfg, 'VOLATILITY_WINDOW', 100)
            # Rolling std via cumsum trick (fast, no pandas)
            self._df_vol = np.zeros(len(df), dtype=np.float64)
            if len(df) > vol_window:
                cumsum = np.cumsum(mid_price)
                cumsum2 = np.cumsum(mid_price ** 2)
                n = vol_window
                mean = (cumsum[n:] - cumsum[:-n]) / n
                mean2 = (cumsum2[n:] - cumsum2[:-n]) / n
                var = mean2 - mean ** 2
                var = np.maximum(var, 0.0)  # numerical safety
                self._df_vol[n:] = np.sqrt(var)
            
            self._df_len = len(df)
            self.max_steps = self._df_len - 1
            logger.info(f"DataFrame mode: {self._df_len:,} rows cached (OFI+OBI+Vol), max_steps={self.max_steps}")
        else:
            self.max_steps = max_steps
        
        # Actions: 0 (Hold), 1 (Buy), 2 (Sell)
        self.action_space = spaces.Discrete(3)
        
        # State: observation space
        low = np.full(cfg.OBS_DIM, -np.inf, dtype=np.float32)
        high = np.full(cfg.OBS_DIM, np.inf, dtype=np.float32)
        low[8] = -1.0
        high[8] = 1.0
        low[10] = 0.0
        high[10] = 1.0
        low[11] = 0.0
        high[11] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # --- Performance: numpy ring buffers instead of deque ---
        self._ofi_buf = np.zeros(self.OFI_LOOKBACK, dtype=np.float64)
        self._ofi_ptr = 0
        self._ofi_count = 0
        
        self._trade_buf = np.zeros(self.OVERTRADE_WINDOW, dtype=np.int8)
        self._trade_ptr = 0
        self._trade_running_sum = 0  # O(1) trade count
        
        self.ofi_ema: float = 0.0
        self.ema_alpha: float = 2.0 / (self.EMA_SPAN + 1)
        
        # Pre-allocated state array (in-place writes, zero allocation per step)
        self.state = np.zeros(cfg.OBS_DIM, dtype=np.float32)
        self.current_position: int = 0
        self.entry_price: float = 0.0
        self.cumulative_reward: float = 0.0
        self.current_step: int = 0
        self.prev_unrealized_pnl: float = 0.0
        self.holding_time: int = 0
        self.prev_spread: float = 0.0
        
        # Latest market data
        self.latest_ofi: float = 0.0
        self.latest_bid: float = 0.0
        self.latest_ask: float = 0.0
        self.latest_spread: float = 0.0
        self.latest_obi: float = 0.0
        self.latest_vol: float = 0.0
        
        # Cache for step()
        self._inv_max_steps = 1.0 / max(self.max_steps, 1)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed, options=options)
        
        self.current_position = 0
        self.entry_price = 0.0
        self.cumulative_reward = 0.0
        self.current_step = 0
        self.prev_unrealized_pnl = 0.0
        self.holding_time = 0
        self.prev_spread = 0.0
        self.ofi_ema = 0.0
        
        self._ofi_buf[:] = 0.0
        self._ofi_ptr = 0
        self._ofi_count = 0
        
        self._trade_buf[:] = 0
        self._trade_ptr = 0
        self._trade_running_sum = 0
        
        self.latest_ofi = 0.0
        self.latest_spread = 0.0
        self.latest_bid = 0.0
        self.latest_ask = 0.0
        self.latest_obi = 0.0
        self.latest_vol = 0.0
        
        self.state[:] = 0.0
        self._inv_max_steps = 1.0 / max(self.max_steps, 1)
        
        logger.debug("Environment reset to flat state.")
        return self.state, {}

    def _write_state(self, unrealized_pnl: float):
        """Writes observation into pre-allocated state array (zero allocation)."""
        s = self.state
        buf = self._ofi_buf
        ptr = self._ofi_ptr
        lb = self.OFI_LOOKBACK
        
        # OFI lookback: read circular buffer in order
        for i in range(lb):
            s[i] = buf[(ptr - lb + i) % lb]
        
        s[5] = self.ofi_ema
        s[6] = self.latest_spread
        s[7] = (self.latest_spread - self.prev_spread) if self.prev_spread > 0 else 0.0
        s[8] = float(self.current_position)
        s[9] = unrealized_pnl
        s[10] = min(self.holding_time * self._inv_max_steps, 1.0)
        s[11] = min(self.current_step * self._inv_max_steps, 1.0)
        s[12] = self.latest_obi
        s[13] = self.latest_vol

    def _push_ofi(self, ofi: float):
        """Appends OFI to circular buffer and updates EMA. O(1)."""
        self._ofi_buf[self._ofi_ptr] = ofi
        self._ofi_ptr = (self._ofi_ptr + 1) % self.OFI_LOOKBACK
        if self._ofi_count < self.OFI_LOOKBACK:
            self._ofi_count += 1
        self.ofi_ema = self.ema_alpha * ofi + (1.0 - self.ema_alpha) * self.ofi_ema

    def _push_trade(self, traded: int):
        """Adds trade flag to ring buffer with O(1) running sum."""
        old_val = self._trade_buf[self._trade_ptr]
        self._trade_running_sum += traded - old_val
        self._trade_buf[self._trade_ptr] = traded
        self._trade_ptr = (self._trade_ptr + 1) % self.OVERTRADE_WINDOW

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes a single step. Performance-optimized inner loop."""
        
        # --- Load market data from numpy cache (inline, no function call) ---
        if self._df_ofi is not None:
            idx = self.current_step if self.current_step < self._df_len else self._df_len - 1
            ofi = self._df_ofi[idx]
            bid = self._df_bid[idx]
            ask = self._df_ask[idx]
            self.latest_ofi = ofi
            self.latest_bid = bid
            self.latest_ask = ask
            self.latest_spread = ask - bid
            self.latest_obi = self._df_obi[idx]
            self.latest_vol = self._df_vol[idx]
            self._push_ofi(ofi)
        
        self.current_step += 1
        self.holding_time += 1
        
        step_reward = 0.0
        trade_executed = False
        realized_pnl = 0.0
        
        # 1. DIFFERENTIAL HOLDING REWARD (ΔPnL)
        entry = self.entry_price
        pos = self.current_position
        
        current_unrealized_pnl = 0.0
        if pos == 1 and entry > 0:
            current_unrealized_pnl = (self.latest_bid - entry) / entry
        elif pos == -1 and entry > 0:
            current_unrealized_pnl = (entry - self.latest_ask) / entry
        
        step_reward = current_unrealized_pnl - self.prev_unrealized_pnl

        # 2. ACTION LOGIC
        if action == 1:  # BUY
            if pos <= 0:
                ep = self.latest_ask
                if ep > 0:
                    step_reward -= self.commission_rate + self.latest_spread * 0.5 / ep
                    if pos == -1 and entry > 0:
                        realized_pnl = (entry - ep) / entry
                        step_reward += realized_pnl
                    self.current_position = 1
                    self.entry_price = ep
                    trade_executed = True
                    self.prev_unrealized_pnl = 0.0
                    self.holding_time = 0
                    self._push_trade(1)
            else:
                step_reward -= self.REDUNDANT_PENALTY
                
        elif action == 2:  # SELL
            if pos >= 0:
                ep = self.latest_bid
                if ep > 0:
                    step_reward -= self.commission_rate + self.latest_spread * 0.5 / ep
                    if pos == 1 and entry > 0:
                        realized_pnl = (ep - entry) / entry
                        step_reward += realized_pnl
                    self.current_position = -1
                    self.entry_price = ep
                    trade_executed = True
                    self.prev_unrealized_pnl = 0.0
                    self.holding_time = 0
                    self._push_trade(1)
            else:
                step_reward -= self.REDUNDANT_PENALTY
        
        # HOLD or after action: push trade=0 if no trade happened
        if not trade_executed:
            self._push_trade(0)
        
        # 3. OVERTRADING PENALTY (O(1) running sum)
        rt = self._trade_running_sum
        if rt > self.OVERTRADE_MAX:
            step_reward -= (rt - self.OVERTRADE_MAX) * self.OVERTRADE_PENALTY
        
        # 4. UPDATE STATE (in-place, zero allocation)
        unrealized_pnl = 0.0
        if self.current_position == 1 and self.entry_price > 0:
            unrealized_pnl = (self.latest_bid - self.entry_price) / self.entry_price
        elif self.current_position == -1 and self.entry_price > 0:
            unrealized_pnl = (self.entry_price - self.latest_ask) / self.entry_price
        
        self.prev_unrealized_pnl = unrealized_pnl
        self.prev_spread = self.latest_spread
        self._write_state(unrealized_pnl)
        self.cumulative_reward += step_reward
        
        info = {
            "pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "position": self.current_position,
            "trade_executed": trade_executed,
            "reward": step_reward,
            "total_trades": self._trade_running_sum,
        }
        
        terminated = unrealized_pnl < cfg.STOP_LOSS_THRESHOLD
        truncated = self.max_steps > 0 and self.current_step >= self.max_steps
            
        return self.state, float(step_reward), terminated, truncated, info

    def update_market_data(self, ofi: float, bid: float, ask: float):
        """
        Injects real-time WebSocket data into the environment.
        Used in live mode (when no DataFrame is provided).
        """
        self.latest_ofi = ofi
        self.latest_bid = bid
        self.latest_ask = ask
        self.latest_spread = ask - bid
        self._push_ofi(ofi)
        
        unrealized_pnl = 0.0
        if self.current_position == 1 and self.entry_price > 0:
            unrealized_pnl = (bid - self.entry_price) / self.entry_price
        elif self.current_position == -1 and self.entry_price > 0:
            unrealized_pnl = (self.entry_price - ask) / self.entry_price
        self._write_state(unrealized_pnl)

    def render(self):
        """Renders the environment to standard output."""
        if self.render_mode == "ansi":
            print(f"Step: {self.current_step} | Pos: {self.current_position} | "
                  f"OFI: {self.latest_ofi:.2f} | Reward: {self.cumulative_reward:.5f}")
