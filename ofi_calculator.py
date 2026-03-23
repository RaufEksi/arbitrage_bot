from typing import Optional
from orderbook import OrderBook
from logger import get_logger

logger = get_logger("OFICalculator")

class OFICalculator:
    """
    Calculates the Order Flow Imbalance (OFI) based on Cont et al. (2014) approach.
    Focuses on the Best Bid and Best Ask (Level-1 OFI).
    """
    
    def __init__(self, orderbook: OrderBook):
        self.orderbook = orderbook
        
        # Track the previous state of the top of the book
        self.prev_best_bid_price: Optional[float] = None
        self.prev_best_bid_qty: Optional[float] = None
        self.prev_best_ask_price: Optional[float] = None
        self.prev_best_ask_qty: Optional[float] = None
        
    def calculate_ofi(self) -> Optional[float]:
        """
        Calculates Cont et al. Level-1 Order Flow Imbalance.
        Must be called continuously as the orderbook updates.
        
        Returns:
            float: The calculated OFI value. Returns None if the orderbook is not ready.
        """
        if not self.orderbook.is_synchronized:
            return None
            
        (best_bid_price, best_bid_qty), (best_ask_price, best_ask_qty) = self.orderbook.get_best_quotes()
        
        # If the orderbook is somehow empty, skip calculation
        if best_bid_qty == 0.0 or best_ask_qty == 0.0:
            return None
        
        # On the very first calculation, we just store the previous state
        if self.prev_best_bid_price is None:
            self._update_prev_state(best_bid_price, best_bid_qty, best_ask_price, best_ask_qty)
            return 0.0
            
        # ---------------------------------------------------------
        # Calculate Bid-side order flow (e)
        # ---------------------------------------------------------
        if best_bid_price > self.prev_best_bid_price:
            # Bid price moved up: old orders eaten/cancelled, new orders added. 
            # Net change at best bid is entirely the new volume.
            e = best_bid_qty
        elif best_bid_price == self.prev_best_bid_price:
            # Bid price stayed the same: update is just the difference in volume.
            e = best_bid_qty - self.prev_best_bid_qty
        else:
            # Bid price moved down: best bid was fully consumed or cancelled.
            e = -self.prev_best_bid_qty
             
        # ---------------------------------------------------------
        # Calculate Ask-side order flow (f)
        # ---------------------------------------------------------
        if best_ask_price < self.prev_best_ask_price:
            # Ask price moved down: new orders added at a lower price.
            # Net change at best ask is entirely the new volume.
            f = best_ask_qty
        elif best_ask_price == self.prev_best_ask_price:
            # Ask price stayed the same: update is just the difference in volume.
            f = best_ask_qty - self.prev_best_ask_qty
        else:
            # Ask price moved up: best ask was fully consumed or cancelled.
            f = -self.prev_best_ask_qty
             
        # ---------------------------------------------------------
        # Overall Level 1 OFI
        # ---------------------------------------------------------
        ofi = e - f
        
        # Update state for the next event
        self._update_prev_state(best_bid_price, best_bid_qty, best_ask_price, best_ask_qty)
        
        return ofi
        
    def _update_prev_state(self, best_bid_price: float, best_bid_qty: float, 
                           best_ask_price: float, best_ask_qty: float):
        """Helper to store current top of book as previous state for the next tick."""
        self.prev_best_bid_price = best_bid_price
        self.prev_best_bid_qty = best_bid_qty
        self.prev_best_ask_price = best_ask_price
        self.prev_best_ask_qty = best_ask_qty
