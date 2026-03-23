import asyncio
from orderbook import OrderBook
from ofi_calculator import OFICalculator
from websocket_manager import BinanceWSManager
from logger import get_logger

logger = get_logger("Main")

async def main():
    logger.info("=========================================")
    logger.info("   Binance HFT L2 Orderbook & OFI Bot")
    logger.info("=========================================")
    
    # Initialize Core System Components
    orderbook = OrderBook()
    ofi_calculator = OFICalculator(orderbook=orderbook)
    ws_manager = BinanceWSManager(orderbook=orderbook, ofi_calculator=ofi_calculator)
    
    logger.info("Components initialized. Entering main event loop.")
    
    # Start the WebSocket connected consumer task
    # This will run indefinitely and handle its own reconnections and state bridging
    try:
        await ws_manager.connect()
    except asyncio.CancelledError:
        logger.info("Shutting down bot gracefully...")
    except Exception as e:
        logger.error(f"Critical execution error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot manually stopped by user (KeyboardInterrupt).")
