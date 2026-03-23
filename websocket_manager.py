import asyncio
import orjson
import websockets
from orderbook import OrderBook
from ofi_calculator import OFICalculator
from config import BINANCE_WS_DEPTH_URL as BINANCE_WS_URL_TEMPLATE, SYMBOL
from logger import get_logger

logger = get_logger("WebSocketClient")

class BinanceWSManager:
    """Manages the WebSocket connection for receiving real-time orderbook depth updates."""
    
    def __init__(self, orderbook: OrderBook, ofi_calculator: OFICalculator):
        self.symbol = SYMBOL.lower()
        self.url = BINANCE_WS_URL_TEMPLATE.format(self.symbol)
        self.orderbook = orderbook
        self.ofi_calculator = ofi_calculator
        
        self.reconnect_delay = 1.0  # Initial delay in seconds
        self.max_reconnect_delay = 60.0

    async def connect(self):
        """Connects to Binance WS and handles automatic reconnection."""
        while True:
            try:
                logger.info(f"Connecting to Binance WebSocket URL: {self.url}")
                async with websockets.connect(self.url) as ws:
                    logger.info("WebSocket connection established.")
                    self.reconnect_delay = 1.0  # Reset reconnect delay on success
                    
                    # Synchronize the orderbook BEFORE processing live data.
                    # Await explicitly to prevent race condition (incomplete book).
                    try:
                        sync_success = await self.orderbook.sync_book()
                        if not sync_success:
                            logger.error("Orderbook snapshot sync returned False. Aborting stream.")
                            continue  # triggers reconnect
                    except Exception as sync_err:
                        logger.error(f"Orderbook sync failed: {sync_err}. Aborting stream.")
                        continue  # triggers reconnect
                    
                    # Orderbook is synchronized — safe to process live stream
                    await self._listen_stream(ws)

            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self.orderbook.is_synchronized = False
                logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                # Exponential backoff
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def _listen_stream(self, ws):
        """Listens for and processes messages from the WebSocket."""
        async for message in ws:
            try:
                event = orjson.loads(message)
                
                # Check if it's a Depth Update Event
                if event.get("e") == "depthUpdate":
                    # process_diff_event applies the update or buffers it.
                    # It returns False if sequence numbers don't match (loss of sync).
                    if self.orderbook.process_diff_event(event):
                        
                        # Only calculate and output OFI if we are fully synchronized
                        if self.orderbook.is_synchronized:
                            ofi = self.ofi_calculator.calculate_ofi()
                            
                            if ofi is not None and ofi != 0.0:
                                # Fetch the best quotes for logging context
                                (bid_price, bid_qty), (ask_price, ask_qty) = self.orderbook.get_best_quotes()
                                spread = ask_price - bid_price
                                
                                # Format OFI output
                                # Highlight positive vs negative mathematically
                                logger.info(
                                    f"OFI: {ofi:>+12.4f} | "
                                    f"Bid: {bid_price:.2f} (vol: {bid_qty:6.2f}) | "
                                    f"Ask: {ask_price:.2f} (vol: {ask_qty:6.2f}) | "
                                    f"Spread: {spread:.2f}"
                                )
                    else:
                        # Orderbook became desynchronized. 
                        # Breaking the listen loop will trigger the outer reconnection logic
                        # which will automatically fetch a new snapshot.
                        logger.warning("Triggering reconnection to resynchronize orderbook...")
                        break
                        
                else:
                    logger.debug(f"Unhandled event: {event}")
                    
            except orjson.JSONDecodeError:
                logger.error("Failed to decode JSON from WebSocket.")
            except Exception as e:
                logger.exception(f"Error processing WebSocket message: {e}")
