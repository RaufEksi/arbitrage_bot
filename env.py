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
    V2 HFT Trading Environment with enriched state space and reward engineering.
    
    Observation Space (12-dim Continuous):
        [0-4]:  OFI lookback window (last 5 ticks)
        [5]:    OFI Exponential Moving Average (EMA-20)
        [6]:    Current Spread
        [7]:    Spread change rate (ΔSpread)
        [8]:    Current Position (-1, 0, 1)
        [9]:    Unrealized PnL
        [10]:   Holding time (normalized by max_steps)
        [11]:   Step progress (current_step / max_steps)
        
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
    
    # --- Reward Tuning Constants (from config.py) ---
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
        
        # --- DataFrame Mode ---
        # When df is provided, the environment reads market data from the DataFrame
        # at each step. When df is None, it works in live/manual mode via update_market_data().
        self.df = df
        if self.df is not None:
            required_cols = {"bid_price", "ask_price", "ofi", "spread"}
            missing = required_cols - set(self.df.columns)
            if missing:
                raise ValueError(f"DataFrame missing required columns: {missing}")
            self.max_steps = len(self.df) - 1  # Auto-detect from data length
            logger.info(f"DataFrame mode: {len(self.df):,} rows loaded, max_steps={self.max_steps}")
        else:
            self.max_steps = max_steps
        
        # Actions: 0 (Hold), 1 (Buy), 2 (Sell)
        self.action_space = spaces.Discrete(3)
        
        # State: observation space
        low = np.full(cfg.OBS_DIM, -np.inf, dtype=np.float32)
        high = np.full(cfg.OBS_DIM, np.inf, dtype=np.float32)
        # Clamp known bounded features
        low[8] = -1.0   # Position
        high[8] = 1.0
        low[10] = 0.0   # Holding time (normalized)
        high[10] = 1.0
        low[11] = 0.0   # Step progress
        high[11] = 1.0
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Internal buffers
        self.ofi_history: deque = deque(maxlen=self.OFI_LOOKBACK)
        self.ofi_ema: float = 0.0
        self.ema_alpha: float = 2.0 / (self.EMA_SPAN + 1)
        self.trade_history: deque = deque(maxlen=self.OVERTRADE_WINDOW)
        
        # State variables
        self.state: np.ndarray = np.zeros(cfg.OBS_DIM, dtype=np.float32)
        self.current_position: int = 0
        self.entry_price: float = 0.0
        self.cumulative_reward: float = 0.0
        self.current_step: int = 0
        self.prev_unrealized_pnl: float = 0.0
        self.holding_time: int = 0       # Ticks since last position change
        self.prev_spread: float = 0.0    # For spread delta
        
        # Latest market data
        self.latest_ofi: float = 0.0
        self.latest_bid: float = 0.0
        self.latest_ask: float = 0.0
        self.latest_spread: float = 0.0
        
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
        
        self.ofi_history.clear()
        self.trade_history.clear()
        for _ in range(self.OFI_LOOKBACK):
            self.ofi_history.append(0.0)
        
        self.latest_ofi = 0.0
        self.latest_spread = 0.0
        self.latest_bid = 0.0
        self.latest_ask = 0.0
        
        self.state = np.zeros(cfg.OBS_DIM, dtype=np.float32)
        
        logger.info("Environment reset to flat state.")
        return self.state, {}

    def _build_state(self, unrealized_pnl: float) -> np.ndarray:
        """Constructs the 12-dimensional observation vector."""
        # OFI lookback (pad with zeros if not enough history)
        ofi_window = list(self.ofi_history)
        
        # Spread change rate
        spread_delta = (self.latest_spread - self.prev_spread) if self.prev_spread > 0 else 0.0
        
        # Normalize holding time and step progress
        max_s = self.max_steps if self.max_steps > 0 else 10000
        holding_norm = min(self.holding_time / max_s, 1.0)
        step_progress = min(self.current_step / max_s, 1.0)
        
        return np.array([
            ofi_window[-5] if len(ofi_window) >= 5 else 0.0,   # [0] OFI t-4
            ofi_window[-4] if len(ofi_window) >= 4 else 0.0,   # [1] OFI t-3
            ofi_window[-3] if len(ofi_window) >= 3 else 0.0,   # [2] OFI t-2
            ofi_window[-2] if len(ofi_window) >= 2 else 0.0,   # [3] OFI t-1
            ofi_window[-1] if len(ofi_window) >= 1 else 0.0,   # [4] OFI t (current)
            self.ofi_ema,                                        # [5] OFI EMA-20
            self.latest_spread,                                  # [6] Spread
            spread_delta,                                        # [7] ΔSpread
            float(self.current_position),                        # [8] Position
            unrealized_pnl,                                      # [9] Unrealized PnL
            holding_norm,                                        # [10] Holding time
            step_progress,                                       # [11] Step progress
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Executes a single step with V2 reward engineering.
        """
        # --- Load market data from DataFrame BEFORE incrementing step ---
        # This ensures step 0 reads df.iloc[0], step 1 reads df.iloc[1], etc.
        if self.df is not None:
            self._load_tick_from_df()
        
        self.current_step += 1
        self.holding_time += 1
        
        prev_position = self.current_position
        step_reward = 0.0
        trade_executed = False
        realized_pnl = 0.0
        
        # ----------------------------------------------------------
        # 1. DIFFERENTIAL HOLDING REWARD (ΔPnL)
        # ----------------------------------------------------------
        current_unrealized_pnl = 0.0
        if prev_position == 1 and self.entry_price > 0:
            current_unrealized_pnl = (self.latest_bid - self.entry_price) / self.entry_price
        elif prev_position == -1 and self.entry_price > 0:
            current_unrealized_pnl = (self.entry_price - self.latest_ask) / self.entry_price
        
        delta_pnl = current_unrealized_pnl - self.prev_unrealized_pnl
        step_reward += delta_pnl

        # ----------------------------------------------------------
        # 2. ACTION LOGIC
        # ----------------------------------------------------------
        if action == 1:  # Request BUY
            if self.current_position <= 0:
                execution_price = self.latest_ask
                if execution_price > 0:
                    slippage_cost = self.latest_spread / 2.0 / execution_price
                    step_reward -= (self.commission_rate + slippage_cost)
                    
                    if self.current_position == -1 and self.entry_price > 0:
                        realized_pnl = (self.entry_price - execution_price) / self.entry_price
                        step_reward += realized_pnl
                    
                    self.current_position = 1
                    self.entry_price = execution_price
                    trade_executed = True
                    self.prev_unrealized_pnl = 0.0
                    self.holding_time = 0
                    self.trade_history.append(1)
            else:
                step_reward -= self.REDUNDANT_PENALTY
                
        elif action == 2:  # Request SELL
            if self.current_position >= 0:
                execution_price = self.latest_bid
                if execution_price > 0:
                    slippage_cost = self.latest_spread / 2.0 / execution_price
                    step_reward -= (self.commission_rate + slippage_cost)
                    
                    if self.current_position == 1 and self.entry_price > 0:
                        realized_pnl = (execution_price - self.entry_price) / self.entry_price
                        step_reward += realized_pnl
                    
                    self.current_position = -1
                    self.entry_price = execution_price
                    trade_executed = True
                    self.prev_unrealized_pnl = 0.0
                    self.holding_time = 0
                    self.trade_history.append(1)
            else:
                step_reward -= self.REDUNDANT_PENALTY
        else:
            # Action == 0 (HOLD)
            self.trade_history.append(0)
        
        # ----------------------------------------------------------
        # 3. REWARD ENGINEERING — V4 SOFT PENALTIES ONLY
        # ----------------------------------------------------------
        
        # Overtrading Penalty: Gentle discouragement of excessive trading
        recent_trades = sum(self.trade_history)
        if recent_trades > self.OVERTRADE_MAX:
            excess = recent_trades - self.OVERTRADE_MAX
            step_reward -= excess * self.OVERTRADE_PENALTY
        
        # ----------------------------------------------------------
        # 4. UPDATE UNREALIZED PnL & STATE
        # ----------------------------------------------------------
        unrealized_pnl = 0.0
        if self.current_position == 1 and self.entry_price > 0:
            unrealized_pnl = (self.latest_bid - self.entry_price) / self.entry_price
        elif self.current_position == -1 and self.entry_price > 0:
            unrealized_pnl = (self.entry_price - self.latest_ask) / self.entry_price
        
        self.prev_unrealized_pnl = unrealized_pnl
        self.prev_spread = self.latest_spread
        
        self.state = self._build_state(unrealized_pnl)
        self.cumulative_reward += step_reward
        
        info = {
            "pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "position": self.current_position,
            "trade_executed": trade_executed,
            "reward": step_reward,
            "total_trades": sum(self.trade_history),
        }
        
        # Termination conditions
        terminated = False
        truncated = False
        if unrealized_pnl < cfg.STOP_LOSS_THRESHOLD:
            terminated = True
            logger.warning("Stop loss triggered! Episode terminated.")
            
        if self.max_steps > 0 and self.current_step >= self.max_steps:
            truncated = True
            
        return self.state, float(step_reward), terminated, truncated, info

    def _load_tick_from_df(self):
        """
        Reads the current tick's market data from the internal DataFrame.
        Called automatically at the start of each step() when in DataFrame mode.
        """
        idx = min(self.current_step, len(self.df) - 1)
        row = self.df.iloc[idx]
        
        ofi = float(row["ofi"])
        bid = float(row["bid_price"])
        ask = float(row["ask_price"])
        
        # Delegate to existing update logic for consistency
        self.update_market_data(ofi=ofi, bid=bid, ask=ask)

    def update_market_data(self, ofi: float, bid: float, ask: float):
        """
        Injects real-time WebSocket data into the environment.
        Updates OFI history, EMA, and recalculates PnL.
        Also called internally by _load_tick_from_df() in DataFrame mode.
        """
        self.latest_ofi = ofi
        self.latest_bid = bid
        self.latest_ask = ask
        self.latest_spread = ask - bid
        
        # Append to OFI history for lookback window
        self.ofi_history.append(ofi)
        
        # Update EMA: EMA_t = alpha * OFI_t + (1-alpha) * EMA_{t-1}
        self.ofi_ema = self.ema_alpha * ofi + (1 - self.ema_alpha) * self.ofi_ema
        
        # Recalculate unrealized PnL
        unrealized_pnl = 0.0
        if self.current_position == 1 and self.entry_price > 0:
            unrealized_pnl = (bid - self.entry_price) / self.entry_price
        elif self.current_position == -1 and self.entry_price > 0:
            unrealized_pnl = (self.entry_price - ask) / self.entry_price
            
        self.state = self._build_state(unrealized_pnl)

    def render(self):
        """Renders the environment to standard output."""
        if self.render_mode == "ansi":
            print(f"Step: {self.current_step} | Pos: {self.current_position} | "
                  f"OFI: {self.latest_ofi:.2f} | Reward: {self.cumulative_reward:.5f}")
