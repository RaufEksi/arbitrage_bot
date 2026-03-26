import asyncio
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
import os
import config as cfg

# Logging Setup
os.system("") # Enable ANSI colors in Windows terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("LiveTrader")

WINDOW_SIZE = 150  # Rolling window required for 100-tick Z-scores
COOLDOWN_TICKS = 50  # Rate-limit: wait 50 ticks after an execution

class LiveTrader:
    def __init__(self, model_path="models/xgb_best.json"):
        # Load XGBoost Multi-Class Classification Model
        self.model = xgb.XGBClassifier()
        try:
            self.model.load_model(model_path)
            logger.info(f"[+] XGBoost AI Yüklendi: {model_path}")
        except Exception as e:
            logger.error(f"[-] XGBoost model okunamadı. Lütfen 'xgb_best.json' dosyasının var olduğundan emin olun. Hata: {e}")
            exit(1)
            
        self.tick_buffer = deque(maxlen=WINDOW_SIZE)
        self.cooldown_counter = 0
        
        # Incremental OFI tracking
        self.prev_bid_price = 0.0
        self.prev_bid_qty = 0.0
        self.prev_ask_price = 0.0
        self.prev_ask_qty = 0.0
        
        self.symbol = cfg.SYMBOL.upper()
        self.ws_url = cfg.TESTNET_WSS_URL.format(cfg.SYMBOL.lower())
        self.time_offset = 0  # To fix Windows/Binance time desync
        
        # PnL & Position Tracker Start
        self.current_position = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.commission_rate = 0.0004  # Testnet/Normal Taker
        self.trade_qty = 0.001  # Risk yönetimi için 0.001 BTC
        
    def _sign_request(self, params: dict):
        query_string = urlencode(params)
        signature = hmac.new(cfg.TESTNET_API_SECRET.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = signature
        return params

    async def _place_order(self, session, side: str, quantity: float):
        """Places a Market Order on Binance Futures Testnet."""
        if not cfg.TESTNET_API_KEY or cfg.TESTNET_API_KEY == "YOUR_TESTNET_API_KEY":
            logger.warning(f"⚠️  SIMULATION (API KEY YOK): {side} {quantity} {self.symbol} Market")
            return
            
        endpoint = f"{cfg.TESTNET_REST_URL}/fapi/v1/order"
        params = {
            "symbol": self.symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": round(quantity, 3),
            "timestamp": int(time.time() * 1000) + self.time_offset
        }
        signed_params = self._sign_request(params)
        headers = {"X-MBX-APIKEY": cfg.TESTNET_API_KEY}
        
        try:
            async with session.post(endpoint, params=signed_params, headers=headers) as resp:
                result = await resp.json()
                if resp.status == 200:
                    logger.info(f"✅ İŞLEM BAŞARILI! Emir ID: {result.get('orderId')}")
                else:
                    logger.error(f"❌ İŞLEM REDDEDİLDİ: {result}")
        except Exception as e:
            logger.error(f"API Ağ Hatası: {e}")

    def calculate_features_and_predict(self, data_list):
        if len(data_list) < 100:
            return None, None
            
        df = pd.DataFrame(data_list)
        
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2.0
        df['spread'] = df['ask_price'] - df['bid_price']
        
        # OBI
        df['obi'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
        
        # EMA
        df['ofi_ema'] = df['ofi'].ewm(span=20, adjust=False).mean()
        df['obi_ema'] = df['obi'].ewm(span=20, adjust=False).mean()
        
        # Rolling Z-Scores (100)
        z_window = 100
        ofi_mean = df['ofi_ema'].rolling(window=z_window, min_periods=1).mean()
        ofi_std = df['ofi_ema'].rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
        df['ofi_z'] = (df['ofi_ema'] - ofi_mean) / ofi_std
        
        obi_mean = df['obi_ema'].rolling(window=z_window, min_periods=1).mean()
        obi_std = df['obi_ema'].rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
        df['obi_z'] = (df['obi_ema'] - obi_mean) / obi_std
        
        # Volatility
        df['volatility'] = df['mid_price'].rolling(window=z_window, min_periods=1).std().fillna(0)
        
        # Extract the state vector for the Model matching exact training order
        features_order = ['ofi', 'obi', 'ofi_ema', 'obi_ema', 'ofi_z', 'obi_z', 'volatility', 'spread']
        last_row = df.iloc[-1]
        
        # Must provide column names to XGBoost to avoid warnings and secure order mapping
        input_data = pd.DataFrame([last_row[features_order].values], columns=features_order)
        
        # Predict
        prediction = self.model.predict(input_data)[0]
        return prediction, last_row

    async def start(self):
        async with aiohttp.ClientSession() as session:
            # Sync Local time with Binance Server Time (Fixes -1021 Timeout errors)
            try:
                logger.info("🕒 Binance sunucu saatiyle senkronize olunuyor...")
                async with session.get(f"{cfg.TESTNET_REST_URL}/fapi/v1/time") as resp:
                    res = await resp.json()
                    server_time = res['serverTime']
                    self.time_offset = server_time - int(time.time() * 1000)
                    logger.info(f"✅ Zaman Sapması (Offset) Hesaplandı: {self.time_offset} ms")
            except Exception as e:
                logger.error(f"Zaman sapması hesabı başarısız: {e}")

            while True:
                try:
                    logger.info(f"Yayın başlatılıyor: {self.ws_url}")
                    async with websockets.connect(self.ws_url) as ws:
                        logger.info("📡 Binance Futures Testnet BookTicker bağlandı. Veri toplanıyor...")
                        
                        while True:
                            message = await ws.recv()
                            data = json.loads(message)
                            
                            # BookTicker filter
                            if 'b' not in data or 'a' not in data:
                                continue
                                
                            bid_price = float(data['b'])
                            bid_qty = float(data['B'])
                            ask_price = float(data['a'])
                            ask_qty = float(data['A'])
                            
                            # Linear L2 OFI Calculation identical to training set
                            ofi = 0.0
                            if self.prev_bid_price > 0:
                                delta_bid_qty = bid_qty if bid_price > self.prev_bid_price else (bid_qty - self.prev_bid_qty if bid_price == self.prev_bid_price else -self.prev_bid_qty)
                                delta_ask_qty = ask_qty if ask_price < self.prev_ask_price else (ask_qty - self.prev_ask_qty if ask_price == self.prev_ask_price else -self.prev_ask_qty)
                                ofi = delta_bid_qty - delta_ask_qty
                            
                            self.prev_bid_price, self.prev_bid_qty = bid_price, bid_qty
                            self.prev_ask_price, self.prev_ask_qty = ask_price, ask_qty
                            
                            row = {
                                "bid_price": bid_price,
                                "bid_qty": bid_qty,
                                "ask_price": ask_price,
                                "ask_qty": ask_qty,
                                "ofi": ofi
                            }
                            self.tick_buffer.append(row)
                            
                            # Enforcement of OverTrading / Aggression Cooldown
                            if self.cooldown_counter > 0:
                                self.cooldown_counter -= 1
                                continue
                                
                            # Start predicting ONLY when the memory buffer is filled (100 ticks)
                            if len(self.tick_buffer) >= 100:
                                pred, features = self.calculate_features_and_predict(list(self.tick_buffer))
                                
                                if pred is not None:
                                    mid = features['mid_price']
                                    ofi_z = features['ofi_z']
                                    
                                    # Output stream on same line to act as an alive indicator without console spam
                                    if len(self.tick_buffer) % 15 == 0:
                                        print(f"| Mid (L2): {mid:.2f} | OFI_Z: {ofi_z:>5.2f} | Model: {pred} |        ", end="\r")
                                    
                                    if pred == 1:
                                        if self.current_position == 1:
                                            # Already long, ignore to prevent over-exposure
                                            continue
                                            
                                        print(f"\n\n🟢 [AL SİNYALİ] XGBoost Z-Kırılımı! Fiyat: {mid:.2f}, OFI_Z: {ofi_z:.2f}")
                                        
                                        order_qty = self.trade_qty
                                        if self.current_position == -1:
                                            # Close Short
                                            comm_close = ask_price * self.trade_qty * self.commission_rate
                                            trade_pnl = ((self.entry_price - ask_price) * self.trade_qty) - comm_close
                                            self.realized_pnl += trade_pnl
                                            self.total_trades += 1
                                            print(f"💸 İŞLEM KAPANDI! Yön: Short | Giriş: {self.entry_price:.2f} | Çıkış: {ask_price:.2f} | İşlem PnL: {trade_pnl:+.4f} USDT | TOPLAM NET PNL: {self.realized_pnl:+.4f} USDT")
                                            order_qty = self.trade_qty * 2  # Close short + Open new long
                                        
                                        # Open Long
                                        comm_open = ask_price * self.trade_qty * self.commission_rate
                                        self.realized_pnl -= comm_open
                                        self.entry_price = ask_price
                                        self.current_position = 1
                                        
                                        await self._place_order(session, "BUY", quantity=order_qty)
                                        self.cooldown_counter = COOLDOWN_TICKS
                                        print(f"⏳ Cooldown devrede ({COOLDOWN_TICKS} tick)...\n")
                                        
                                    elif pred == 2:
                                        if self.current_position == -1:
                                            # Already short, ignore
                                            continue
                                            
                                        print(f"\n\n🔴 [SAT SİNYALİ] XGBoost Z-Kırılımı! Fiyat: {mid:.2f}, OFI_Z: {ofi_z:.2f}")
                                        
                                        order_qty = self.trade_qty
                                        if self.current_position == 1:
                                            # Close Long
                                            comm_close = bid_price * self.trade_qty * self.commission_rate
                                            trade_pnl = ((bid_price - self.entry_price) * self.trade_qty) - comm_close
                                            self.realized_pnl += trade_pnl
                                            self.total_trades += 1
                                            print(f"💸 İŞLEM KAPANDI! Yön: Long | Giriş: {self.entry_price:.2f} | Çıkış: {bid_price:.2f} | İşlem PnL: {trade_pnl:+.4f} USDT | TOPLAM NET PNL: {self.realized_pnl:+.4f} USDT")
                                            order_qty = self.trade_qty * 2  # Close long + Open new short
                                            
                                        # Open Short
                                        comm_open = bid_price * self.trade_qty * self.commission_rate
                                        self.realized_pnl -= comm_open
                                        self.entry_price = bid_price
                                        self.current_position = -1
                                        
                                        await self._place_order(session, "SELL", quantity=order_qty)
                                        self.cooldown_counter = COOLDOWN_TICKS
                                        print(f"⏳ Cooldown devrede ({COOLDOWN_TICKS} tick)...\n")
                                        
                                    elif pred == 0:
                                        # Hold
                                        pass
                                        
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WS Bağlantısı koptu. 3 saniye içinde yeniden deneniyor...")
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.error(f"Beklenmeyen Hata: {e}")
                    await asyncio.sleep(3)

if __name__ == "__main__":
    trader = LiveTrader()
    try:
        asyncio.run(trader.start())
    except KeyboardInterrupt:
        print("\n" + "="*50)
        print(f"🛑 Bot Durduruluyor... Toplam İşlem: {trader.total_trades}, Nihai Net PnL: {trader.realized_pnl:+.4f} USDT")
        print("="*50 + "\n")
