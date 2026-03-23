"""
data_collector.py -- Production-ready Binance L2 BookTicker Data Collector

Connects to Binance WebSocket bookTicker stream, computes real-time OFI
and Spread, and saves the data to CSV for offline RL training and backtesting.

Usage:
    python data_collector.py                    # Default: BTCUSDT, 10000 ticks
    python data_collector.py --symbol ETHUSDT --ticks 50000
"""

import os
import sys
import time
import json
import asyncio
import signal
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    import websockets
except ImportError:
    print("websockets not installed. Run: pip install websockets")
    sys.exit(1)

from logger import get_logger

logger = get_logger("DataCollector")

# ============================================================
# Configuration
# ============================================================
DATA_DIR = "./data/"
DEFAULT_SYMBOL = "btcusdt"
DEFAULT_TICKS = 10_000
SAVE_INTERVAL = 10_000          # Save to CSV every N ticks
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/{}@bookTicker"
RECONNECT_DELAY_SEC = 3
MAX_RECONNECT_ATTEMPTS = 10
TPS_LOG_INTERVAL_SEC = 5        # Log ticks-per-second every N seconds


class OFIDataCollector:
    """
    Asynchronous Binance bookTicker data collector with real-time OFI calculation.
    
    The bookTicker stream provides the best bid/ask price and quantity updates
    in near real-time (~10ms latency). OFI is computed per tick as:
        OFI_t = delta(bid_qty) - delta(ask_qty)
    """
    
    def __init__(self, symbol: str, max_ticks: int, save_interval: int):
        self.symbol = symbol.lower()
        self.max_ticks = max_ticks
        self.save_interval = save_interval
        self.ws_url = BINANCE_WS_URL.format(self.symbol)
        
        # OFI tracking state
        self.prev_bid_price: float = 0.0
        self.prev_bid_qty: float = 0.0
        self.prev_ask_price: float = 0.0
        self.prev_ask_qty: float = 0.0
        self.is_first_tick: bool = True
        
        # Data storage
        self.records: list = []
        self.tick_count: int = 0
        
        # TPS tracking
        self.tps_start_time: float = time.monotonic()
        self.tps_tick_counter: int = 0
        
        # Graceful shutdown
        self._shutdown_event = asyncio.Event()
        
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def _compute_ofi(self, bid_price: float, bid_qty: float,
                     ask_price: float, ask_qty: float) -> float:
        """
        Computes Order Flow Imbalance using Cont et al. (2014) methodology.
        
        OFI = delta_bid_volume - delta_ask_volume
        
        Where delta_bid_volume accounts for price level changes:
        - If bid price increases: new liquidity arrived -> delta = +bid_qty
        - If bid price decreases: liquidity withdrawn  -> delta = -prev_bid_qty
        - If bid price unchanged: delta = bid_qty - prev_bid_qty
        
        Symmetric logic for ask side (inverted).
        """
        if self.is_first_tick:
            self.prev_bid_price = bid_price
            self.prev_bid_qty = bid_qty
            self.prev_ask_price = ask_price
            self.prev_ask_qty = ask_qty
            self.is_first_tick = False
            return 0.0
        
        # Bid side delta
        if bid_price > self.prev_bid_price:
            delta_bid = bid_qty
        elif bid_price < self.prev_bid_price:
            delta_bid = -self.prev_bid_qty
        else:
            delta_bid = bid_qty - self.prev_bid_qty
        
        # Ask side delta
        if ask_price < self.prev_ask_price:
            delta_ask = ask_qty
        elif ask_price > self.prev_ask_price:
            delta_ask = -self.prev_ask_qty
        else:
            delta_ask = ask_qty - self.prev_ask_qty
        
        # Store for next tick
        self.prev_bid_price = bid_price
        self.prev_bid_qty = bid_qty
        self.prev_ask_price = ask_price
        self.prev_ask_qty = ask_qty
        
        return delta_bid - delta_ask
    
    def _log_tps(self):
        """Logs ticks-per-second at configured intervals."""
        now = time.monotonic()
        elapsed = now - self.tps_start_time
        if elapsed >= TPS_LOG_INTERVAL_SEC:
            tps = self.tps_tick_counter / elapsed
            logger.info(
                f"TPS: {tps:.1f} ticks/sec | "
                f"Collected: {self.tick_count:,}/{self.max_ticks:,} "
                f"({self.tick_count / self.max_ticks * 100:.1f}%)"
            )
            self.tps_start_time = now
            self.tps_tick_counter = 0
    
    def _save_to_csv(self, final: bool = False):
        """Saves accumulated records to CSV."""
        if not self.records:
            return
        
        filename = f"{self.symbol}_ofi_data.csv"
        filepath = os.path.join(DATA_DIR, filename)
        
        df = pd.DataFrame(self.records)
        
        # Append if file exists, otherwise create with header
        if os.path.exists(filepath) and not final:
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            # On final save, reload everything and write clean
            if final and os.path.exists(filepath):
                existing = pd.read_csv(filepath)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_csv(filepath, index=False)
        
        saved_count = len(self.records)
        self.records.clear()
        
        tag = "FINAL" if final else "checkpoint"
        logger.info(f"[{tag}] Saved {saved_count:,} ticks -> {filepath} "
                     f"(total on disk: {self.tick_count:,})")
    
    def _process_tick(self, data: dict):
        """Processes a single bookTicker message."""
        try:
            bid_price = float(data["b"])   # Best bid price
            bid_qty = float(data["B"])     # Best bid qty
            ask_price = float(data["a"])   # Best ask price
            ask_qty = float(data["A"])     # Best ask qty
        except (KeyError, ValueError) as e:
            logger.warning(f"Malformed tick data: {e}")
            return
        
        spread = ask_price - bid_price
        ofi = self._compute_ofi(bid_price, bid_qty, ask_price, ask_qty)
        mid_price = (bid_price + ask_price) / 2.0
        
        self.records.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bid_price": bid_price,
            "bid_qty": bid_qty,
            "ask_price": ask_price,
            "ask_qty": ask_qty,
            "mid_price": mid_price,
            "spread": spread,
            "ofi": ofi,
        })
        
        self.tick_count += 1
        self.tps_tick_counter += 1
        
        # Periodic save
        if self.tick_count % self.save_interval == 0:
            self._save_to_csv()
        
        # TPS logging
        self._log_tps()
    
    async def _connect_and_collect(self):
        """Main WebSocket connection loop with auto-reconnect."""
        attempt = 0
        
        while not self._shutdown_event.is_set() and self.tick_count < self.max_ticks:
            try:
                logger.info(f"Connecting to {self.ws_url}...")
                async with websockets.connect(self.ws_url, ping_interval=20) as ws:
                    attempt = 0  # Reset on successful connection
                    logger.info(f"Connected. Collecting {self.symbol.upper()} bookTicker data...")
                    
                    async for message in ws:
                        if self._shutdown_event.is_set():
                            break
                        if self.tick_count >= self.max_ticks:
                            break
                        
                        data = json.loads(message)
                        self._process_tick(data)
                        
            except websockets.exceptions.ConnectionClosed as e:
                attempt += 1
                logger.warning(f"Connection closed: {e}. Reconnect attempt {attempt}/{MAX_RECONNECT_ATTEMPTS}")
            except Exception as e:
                attempt += 1
                logger.error(f"WebSocket error: {e}. Reconnect attempt {attempt}/{MAX_RECONNECT_ATTEMPTS}")
            
            if attempt >= MAX_RECONNECT_ATTEMPTS:
                logger.error("Max reconnection attempts reached. Saving data and exiting.")
                break
            
            if not self._shutdown_event.is_set() and self.tick_count < self.max_ticks:
                logger.info(f"Reconnecting in {RECONNECT_DELAY_SEC}s...")
                await asyncio.sleep(RECONNECT_DELAY_SEC)
    
    async def run(self):
        """Entry point for the collector."""
        logger.info("=" * 50)
        logger.info(f"Binance Data Collector")
        logger.info(f"Symbol     : {self.symbol.upper()}")
        logger.info(f"Target     : {self.max_ticks:,} ticks")
        logger.info(f"Save every : {self.save_interval:,} ticks")
        logger.info(f"Output     : {DATA_DIR}{self.symbol}_ofi_data.csv")
        logger.info("=" * 50)
        
        # Register signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass
        
        try:
            await self._connect_and_collect()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            # Final save on exit
            self._save_to_csv(final=True)
            logger.info(f"Collection complete. Total ticks: {self.tick_count:,}")


def main():
    parser = argparse.ArgumentParser(description="Binance L2 BookTicker Data Collector")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL,
                        help=f"Trading pair (default: {DEFAULT_SYMBOL})")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS,
                        help=f"Number of ticks to collect (default: {DEFAULT_TICKS:,})")
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL,
                        help=f"Save interval in ticks (default: {SAVE_INTERVAL:,})")
    args = parser.parse_args()
    
    collector = OFIDataCollector(
        symbol=args.symbol,
        max_ticks=args.ticks,
        save_interval=args.save_interval,
    )
    
    asyncio.run(collector.run())


if __name__ == "__main__":
    main()
