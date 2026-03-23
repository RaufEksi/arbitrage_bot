import logging

#=================================================
# Binance API Configuration
#=================================================

# REST API Base URL for Snapshot
# Use testnet or live depending on your needs. Live is better for OFI data.
BINANCE_REST_URL = "https://api.binance.com/api/v3/depth"

# WebSocket Stream URL setup
# Using diff.depth stream for orderbook updates
BINANCE_WS_URL_TEMPLATE = "wss://stream.binance.com:9443/ws/{}@depth@100ms"

#=================================================
# Bot Configuration
#=================================================

# Target trading pair
SYMBOL = "btcusdt"

# Depth level for the initial snapshot (valid limits: 5, 10, 20, 50, 100, 500, 1000, 5000)
# A deeper snapshot allows for handling sudden price movements, 1000 is safe.
SNAPSHOT_LIMIT = 1000

# Optional: Number of levels to output on console
DISPLAY_LEVELS = 5

# Logging level
LOG_LEVEL = logging.INFO
