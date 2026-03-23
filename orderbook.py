import aiohttp
from typing import Dict, Any, Tuple
from config import BINANCE_REST_URL, SYMBOL, SNAPSHOT_LIMIT
from logger import get_logger

logger = get_logger("OrderBook")

class OrderBook:
    """
    Maintains the local L2 Order Book state by first fetching a snapshot 
    and then carefully applying depth updates.
    """
    
    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol.upper()
        self.bids: Dict[float, float] = {}  # price -> volume
        self.asks: Dict[float, float] = {}  # price -> volume
        
        self.last_update_id: int = 0
        self.is_synchronized: bool = False
        
        # Buffer to keep incoming WS events before the snapshot is successfully fetched
        self.event_buffer: list = []
        
        # Tracks the last sequence 'u' (Final update ID in event) processed
        self.last_processed_u: int = 0

    async def fetch_snapshot(self) -> bool:
        """
        Fetches the initial L2 Orderbook Snapshot via Binance REST API.
        """
        url = f"{BINANCE_REST_URL}?symbol={self.symbol}&limit={SNAPSHOT_LIMIT}"
        logger.info(f"Fetching orderbook snapshot from {url}")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch snapshot: HTTP {response.status}")
                        return False
                    
                    data = await response.json()
                    self.last_update_id = data['lastUpdateId']
                    
                    self.bids.clear()
                    self.asks.clear()
                    
                    for price_str, qty_str in data['bids']:
                        self.bids[float(price_str)] = float(qty_str)
                    
                    for price_str, qty_str in data['asks']:
                        self.asks[float(price_str)] = float(qty_str)
                        
                    logger.info(f"Snapshot successfully received. LastUpdateId: {self.last_update_id}")
                    return True
        except Exception as e:
            logger.error(f"Error fetching snapshot: {e}")
            return False

    async def sync_book(self) -> bool:
        """
        Coordinates fetching the snapshot and processing any buffered events 
        to ensure the orderbook is fully synchronized with the WebSocket stream.
        """
        self.is_synchronized = False
        self.last_processed_u = 0
        self.last_update_id = 0
        
        success = await self.fetch_snapshot()
        if not success:
            return False
            
        # Process buffered events that came in while waiting for the snapshot
        logger.info(f"Processing buffered events... Buffer size: {len(self.event_buffer)}")
        
        buffer_copy = self.event_buffer.copy()
        self.event_buffer.clear()
        
        for event in buffer_copy:
            if not self.process_diff_event(event):
                self.event_buffer.clear()
                return False
                
        if self.is_synchronized:
            logger.info("OrderBook is synchronized with the WebSocket stream (via buffer).")
        else:
            logger.info("Snapshot fetched. Waiting for the first valid stream event to synchronize...")
        return True

    def process_diff_event(self, event: Dict[str, Any]) -> bool:
        """
        Process incoming depth Update Event from WebSocket. 
        Returns True if the orderbook is updated successfully or event is buffered.
        Returns False if a sequence gap is detected (out of sync).
        """
        U = event['U']  # First update ID in event
        u = event['u']  # Final update ID in event
        
        # Step 1: Buffer events if snapshot is not fetched yet
        if self.last_update_id == 0:
            self.event_buffer.append(event)
            return True
            
        # Step 2: Look for the first valid event after snapshot
        if self.last_processed_u == 0:
            if u <= self.last_update_id:
                # Drop any event where u is <= lastUpdateId in the snapshot
                return True
                
            if U <= self.last_update_id + 1 and u >= self.last_update_id + 1:
                # Found the first valid event
                self.last_processed_u = u
                self._apply_updates(event['b'], event['a'])
                self.is_synchronized = True
                return True
            else:
                logger.warning(f"Failed to sync with first event. Expected U<={self.last_update_id + 1} and u>={self.last_update_id + 1}, got U={U}, u={u}")
                self.is_synchronized = False
                return False
                
        # Step 3: Normal processing after synchronization
        if U != self.last_processed_u + 1:
            logger.warning(f"Orderbook out of sync. Expected U={self.last_processed_u + 1}, got {U}. Re-sync required.")
            self.is_synchronized = False
            return False
            
        # Apply updates safely
        self._apply_updates(event['b'], event['a'])
        self.last_processed_u = u
        return True
        
    def _apply_updates(self, bids_data: list, asks_data: list):
        """
        Applies a list of bids and asks to the local book.
        A quantity of 0 means the price level should be removed.
        """
        for price_str, qty_str in bids_data:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
                
        for price_str, qty_str in asks_data:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    def get_best_bid(self) -> Tuple[float, float]:
        """Returns (best_bid_price, best_bid_qty)"""
        if not self.bids:
            return 0.0, 0.0
        best_price = max(self.bids.keys())
        return best_price, self.bids[best_price]

    def get_best_ask(self) -> Tuple[float, float]:
        """Returns (best_ask_price, best_ask_qty)"""
        if not self.asks:
            return float('inf'), 0.0
        best_price = min(self.asks.keys())
        return best_price, self.asks[best_price]
        
    def get_best_quotes(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Returns ((best_bid_price, best_bid_qty), (best_ask_price, best_ask_qty))"""
        return self.get_best_bid(), self.get_best_ask()
