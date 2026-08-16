"""
live_trader.py -- Asynchronous Live Trading Engine for Binance Futures Testnet.

Connects to the bookTicker WebSocket stream, computes real-time OFI/OBI features
with rolling Z-scores, and executes market orders based on XGBoost predictions.

Usage:
    python live_trader.py
"""

import asyncio
import sys
import websockets
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import deque
import time
import hmac
import hashlib
import logging
from urllib.parse import urlencode
import aiohttp

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveTrader")


class LiveTrader:
    """
    XGBoost-powered live trading engine for Binance Futures.

    Ingests real-time bookTicker data via WebSocket, engineers micro-market
    features (OFI/OBI Z-scores, volatility, spread) on a sliding window,
    and triggers market orders when directional breakouts are detected.
    """

    def __init__(self, model_path: str = "models/xgb_best.json"):
        # Load XGBoost multi-class classification model
        self.model = xgb.XGBClassifier()
        try:
            self.model.load_model(model_path)
            logger.info(f"[+] XGBoost model loaded: {model_path}")
        except Exception as e:
            logger.error(
                f"[-] Failed to load XGBoost model. "
                f"Ensure '{model_path}' exists. Error: {e}"
            )
            sys.exit(1)

        self.tick_buffer: deque = deque(maxlen=cfg.WINDOW_SIZE)
        self.cooldown_counter: int = 0

        # Incremental OFI tracking
        self.prev_bid_price: float = 0.0
        self.prev_bid_qty: float = 0.0
        self.prev_ask_price: float = 0.0
        self.prev_ask_qty: float = 0.0

        self.symbol: str = cfg.SYMBOL.upper()
        self.ws_url: str = cfg.TESTNET_WSS_URL.format(cfg.SYMBOL.lower())
        self.time_offset: int = 0  # Binance server time drift correction

        # PnL & position tracking
        self.current_position: int = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.total_trades: int = 0
        self.commission_rate: float = cfg.TAKER_COMMISSION
        self.trade_qty: float = cfg.TRADE_QTY

    def _sign_request(self, params: dict) -> dict:
        """Signs API request parameters with HMAC-SHA256."""
        query_string = urlencode(params)
        signature = hmac.new(
            cfg.TESTNET_API_SECRET.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    async def _place_order(self, session: aiohttp.ClientSession, side: str, quantity: float) -> None:
        """Places a market order on Binance Futures Testnet."""
        if not cfg.TESTNET_API_KEY:
            logger.warning(f"SIMULATION (no API key): {side} {quantity} {self.symbol} MARKET")
            return

        endpoint = f"{cfg.TESTNET_REST_URL}/fapi/v1/order"
        params = {
            "symbol": self.symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": round(quantity, 3),
            "timestamp": int(time.time() * 1000) + self.time_offset,
        }
        signed_params = self._sign_request(params)
        headers = {"X-MBX-APIKEY": cfg.TESTNET_API_KEY}

        try:
            async with session.post(endpoint, params=signed_params, headers=headers) as resp:
                result = await resp.json()
                if resp.status == 200:
                    logger.info(f"ORDER FILLED | Order ID: {result.get('orderId')}")
                else:
                    logger.error(f"ORDER REJECTED | {result}")
        except Exception as e:
            logger.error(f"API network error: {e}")

    def calculate_features_and_predict(self, data_list: list) -> tuple:
        """
        Computes feature vector from tick buffer and runs XGBoost prediction.

        Returns:
            (prediction, last_row) or (None, None) if insufficient data.
        """
        if len(data_list) < 100:
            return None, None

        df = pd.DataFrame(data_list)

        df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2.0
        df["spread"] = df["ask_price"] - df["bid_price"]

        # OBI (Order Book Imbalance)
        df["obi"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"] + 1e-8)

        # Exponential Moving Averages
        df["ofi_ema"] = df["ofi"].ewm(span=20, adjust=False).mean()
        df["obi_ema"] = df["obi"].ewm(span=20, adjust=False).mean()

        # Rolling Z-Scores (window=100)
        z_window = 100
        ofi_mean = df["ofi_ema"].rolling(window=z_window, min_periods=1).mean()
        ofi_std = df["ofi_ema"].rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
        df["ofi_z"] = (df["ofi_ema"] - ofi_mean) / ofi_std

        obi_mean = df["obi_ema"].rolling(window=z_window, min_periods=1).mean()
        obi_std = df["obi_ema"].rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
        df["obi_z"] = (df["obi_ema"] - obi_mean) / obi_std

        # Volatility
        df["volatility"] = df["mid_price"].rolling(window=z_window, min_periods=1).std().fillna(0)

        # Extract feature vector matching exact training column order
        features_order = ["ofi", "obi", "ofi_ema", "obi_ema", "ofi_z", "obi_z", "volatility", "spread"]
        last_row = df.iloc[-1]

        input_data = pd.DataFrame([last_row[features_order].values], columns=features_order)
        prediction = self.model.predict(input_data)[0]
        return prediction, last_row

    def _log_trade_close(self, direction: str, entry: float, exit_price: float, pnl: float) -> None:
        """Logs a closed trade with PnL details."""
        logger.info(
            f"TRADE CLOSED | Direction: {direction} | "
            f"Entry: {entry:.2f} | Exit: {exit_price:.2f} | "
            f"Trade PnL: {pnl:+.4f} USDT | Total PnL: {self.realized_pnl:+.4f} USDT"
        )

    async def start(self) -> None:
        """Main event loop: connects to WebSocket and processes tick data."""
        async with aiohttp.ClientSession() as session:
            # Sync local clock with Binance server time (prevents -1021 errors)
            try:
                logger.info("Synchronizing with Binance server clock...")
                async with session.get(f"{cfg.TESTNET_REST_URL}/fapi/v1/time") as resp:
                    res = await resp.json()
                    server_time = res["serverTime"]
                    self.time_offset = server_time - int(time.time() * 1000)
                    logger.info(f"Time offset calculated: {self.time_offset} ms")
            except Exception as e:
                logger.error(f"Time sync failed: {e}")

            while True:
                try:
                    logger.info(f"Connecting to stream: {self.ws_url}")
                    async with websockets.connect(self.ws_url) as ws:
                        logger.info("Connected to Binance Futures Testnet bookTicker stream.")

                        while True:
                            message = await ws.recv()
                            data = json.loads(message)

                            # Filter bookTicker messages
                            if "b" not in data or "a" not in data:
                                continue

                            bid_price = float(data["b"])
                            bid_qty = float(data["B"])
                            ask_price = float(data["a"])
                            ask_qty = float(data["A"])

                            # Compute OFI identical to training pipeline
                            ofi = 0.0
                            if self.prev_bid_price > 0:
                                delta_bid_qty = (
                                    bid_qty if bid_price > self.prev_bid_price
                                    else (bid_qty - self.prev_bid_qty if bid_price == self.prev_bid_price
                                          else -self.prev_bid_qty)
                                )
                                delta_ask_qty = (
                                    ask_qty if ask_price < self.prev_ask_price
                                    else (ask_qty - self.prev_ask_qty if ask_price == self.prev_ask_price
                                          else -self.prev_ask_qty)
                                )
                                ofi = delta_bid_qty - delta_ask_qty

                            self.prev_bid_price, self.prev_bid_qty = bid_price, bid_qty
                            self.prev_ask_price, self.prev_ask_qty = ask_price, ask_qty

                            row = {
                                "bid_price": bid_price,
                                "bid_qty": bid_qty,
                                "ask_price": ask_price,
                                "ask_qty": ask_qty,
                                "ofi": ofi,
                            }
                            self.tick_buffer.append(row)

                            # Enforce cooldown after execution
                            if self.cooldown_counter > 0:
                                self.cooldown_counter -= 1
                                continue

                            # Predict only when buffer has enough data
                            if len(self.tick_buffer) >= 100:
                                pred, features = self.calculate_features_and_predict(list(self.tick_buffer))

                                if pred is not None:
                                    mid = features["mid_price"]
                                    ofi_z = features["ofi_z"]

                                    # Periodic status output (alive indicator)
                                    if len(self.tick_buffer) % 15 == 0:
                                        print(
                                            f"| Mid: {mid:.2f} | OFI_Z: {ofi_z:>5.2f} | Pred: {pred} |        ",
                                            end="\r",
                                        )

                                    if pred == 1:
                                        if self.current_position == 1:
                                            continue  # Already long

                                        logger.info(f"BUY SIGNAL | Z-Score breakout | Price: {mid:.2f}, OFI_Z: {ofi_z:.2f}")

                                        order_qty = self.trade_qty
                                        if self.current_position == -1:
                                            # Close short position
                                            comm_close = ask_price * self.trade_qty * self.commission_rate
                                            trade_pnl = ((self.entry_price - ask_price) * self.trade_qty) - comm_close
                                            self.realized_pnl += trade_pnl
                                            self.total_trades += 1
                                            self._log_trade_close("SHORT", self.entry_price, ask_price, trade_pnl)
                                            order_qty = self.trade_qty * 2  # Close short + open long

                                        # Open long
                                        comm_open = ask_price * self.trade_qty * self.commission_rate
                                        self.realized_pnl -= comm_open
                                        self.entry_price = ask_price
                                        self.current_position = 1

                                        await self._place_order(session, "BUY", quantity=order_qty)
                                        self.cooldown_counter = cfg.COOLDOWN_TICKS
                                        logger.info(f"Cooldown active ({cfg.COOLDOWN_TICKS} ticks)")

                                    elif pred == 2:
                                        if self.current_position == -1:
                                            continue  # Already short

                                        logger.info(f"SELL SIGNAL | Z-Score breakout | Price: {mid:.2f}, OFI_Z: {ofi_z:.2f}")

                                        order_qty = self.trade_qty
                                        if self.current_position == 1:
                                            # Close long position
                                            comm_close = bid_price * self.trade_qty * self.commission_rate
                                            trade_pnl = ((bid_price - self.entry_price) * self.trade_qty) - comm_close
                                            self.realized_pnl += trade_pnl
                                            self.total_trades += 1
                                            self._log_trade_close("LONG", self.entry_price, bid_price, trade_pnl)
                                            order_qty = self.trade_qty * 2  # Close long + open short

                                        # Open short
                                        comm_open = bid_price * self.trade_qty * self.commission_rate
                                        self.realized_pnl -= comm_open
                                        self.entry_price = bid_price
                                        self.current_position = -1

                                        await self._place_order(session, "SELL", quantity=order_qty)
                                        self.cooldown_counter = cfg.COOLDOWN_TICKS
                                        logger.info(f"Cooldown active ({cfg.COOLDOWN_TICKS} ticks)")

                                    elif pred == 0:
                                        pass  # Hold

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection lost. Reconnecting in 3s...")
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    await asyncio.sleep(3)


if __name__ == "__main__":
    trader = LiveTrader()
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print(
            f"Bot stopped. Total trades: {trader.total_trades}, "
            f"Net PnL: {trader.realized_pnl:+.4f} USDT"
        )
        print("=" * 50 + "\n")
