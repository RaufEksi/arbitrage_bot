"""
download_historical.py -- Bulk download Binance historical trade data

Downloads aggTrades from Binance REST API in paginated batches,
reconstructs approximate bookTicker data (best bid/ask), and saves as CSV.

Millions of trades in minutes — much faster than websocket collection.

Usage:
    python download_historical.py                      # 500K trades (default)
    python download_historical.py --trades 1000000     # 1M trades
    python download_historical.py --hours 24           # Last 24 hours of trades
"""

import os
import sys
import time
import argparse

import requests
import numpy as np
import pandas as pd

import config as cfg
from logger import get_logger

logger = get_logger("HistoricalDownloader")

SYMBOL = cfg.SYMBOL.upper()
AGG_TRADES_URL = "https://data-api.binance.vision/api/v3/aggTrades"
BOOK_TICKER_URL = "https://data-api.binance.vision/api/v3/ticker/bookTicker"
DEPTH_URL = "https://data-api.binance.vision/api/v3/depth"
MAX_PER_REQUEST = 1000
RATE_LIMIT_SLEEP = 0.1  # 100ms between requests (safe for 1200 req/min limit)


def get_current_book():
    """Gets current best bid/ask from Binance."""
    resp = requests.get(BOOK_TICKER_URL, params={"symbol": SYMBOL}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return float(data["bidPrice"]), float(data["bidQty"]), float(data["askPrice"]), float(data["askQty"])


def download_agg_trades(target_trades: int = 500_000, hours_back: int = None):
    """
    Downloads aggTrades from Binance REST API.
    Returns DataFrame with columns: timestamp, price, qty, isBuyerMaker.
    """
    all_records = []
    total = 0
    
    # Start from now and go backwards, or from hours_back
    if hours_back:
        end_time = int(time.time() * 1000)
        start_time = end_time - (hours_back * 3600 * 1000)
        params = {"symbol": SYMBOL, "startTime": start_time, "endTime": end_time, "limit": MAX_PER_REQUEST}
    else:
        params = {"symbol": SYMBOL, "limit": MAX_PER_REQUEST}
    
    logger.info(f"Downloading {target_trades:,} aggTrades for {SYMBOL}...")
    
    start = time.time()
    last_id = None
    
    while total < target_trades:
        if last_id is not None:
            params["fromId"] = last_id + 1
            # Remove time params when using fromId
            params.pop("startTime", None)
            params.pop("endTime", None)
        
        try:
            resp = requests.get(AGG_TRADES_URL, params=params, timeout=15)
            resp.raise_for_status()
            trades = resp.json()
        except requests.RequestException as e:
            logger.warning(f"Request failed: {e}, retrying in 1s...")
            time.sleep(1)
            continue
        
        if not trades:
            # Go backwards from the earliest trade we have
            if all_records:
                earliest_id = all_records[0]["a"]
                params = {"symbol": SYMBOL, "fromId": max(earliest_id - MAX_PER_REQUEST, 0), "limit": MAX_PER_REQUEST}
                continue
            break
        
        all_records.extend(trades)
        total = len(all_records)
        last_id = trades[-1]["a"]  # last aggTrade ID
        
        elapsed = time.time() - start
        tps = total / elapsed if elapsed > 0 else 0
        pct = min(total / target_trades * 100, 100)
        
        if total % 50000 < MAX_PER_REQUEST:
            logger.info(f"  {total:>10,} / {target_trades:,} ({pct:.1f}%) | {tps:,.0f} trades/sec")
        
        time.sleep(RATE_LIMIT_SLEEP)
    
    elapsed = time.time() - start
    logger.info(f"Downloaded {total:,} trades in {elapsed:.1f}s ({total/elapsed:,.0f} trades/sec)")
    
    return all_records


def trades_to_book_ticks(trades: list) -> pd.DataFrame:
    """
    Converts aggTrades into approximate bookTicker format.
    
    Strategy:
    - Buyer-maker trade (isBuyerMaker=True) = sell aggressor → price ≈ bid
    - Seller-maker trade (isBuyerMaker=False) = buy aggressor → price ≈ ask
    - Track running best bid/ask from trade flow
    """
    logger.info("Converting trades to bookTicker format...")
    
    records = []
    last_bid = 0.0
    last_ask = 0.0
    last_bid_qty = 0.0
    last_ask_qty = 0.0
    
    for t in trades:
        price = float(t["p"])
        qty = float(t["q"])
        ts = t["T"]  # trade time in ms
        
        if t["m"]:  # isBuyerMaker = True → seller aggressor → trade at bid
            last_bid = price
            last_bid_qty = qty
        else:  # buyer aggressor → trade at ask
            last_ask = price
            last_ask_qty = qty
        
        # Only emit ticks where we have both bid and ask
        if last_bid > 0 and last_ask > 0 and last_bid < last_ask:
            records.append({
                "timestamp": pd.Timestamp(ts, unit="ms", tz="UTC"),
                "bid_price": last_bid,
                "bid_qty": last_bid_qty,
                "ask_price": last_ask,
                "ask_qty": last_ask_qty,
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Converted {len(trades):,} trades -> {len(df):,} book ticks")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes OFI, spread, mid_price from book tick data."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2.0
    df["spread"] = df["ask_price"] - df["bid_price"]
    
    # OFI = ΔBidQty - ΔAskQty
    delta_bid = df["bid_qty"].diff().fillna(0)
    delta_ask = df["ask_qty"].diff().fillna(0)
    df["ofi"] = delta_bid - delta_ask
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Download Binance historical trade data")
    parser.add_argument("--trades", type=int, default=500_000,
                        help="Number of aggTrades to download (default: 500,000)")
    parser.add_argument("--hours", type=int, default=None,
                        help="Download last N hours of trades")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()
    
    output_path = args.output or cfg.DATA_PATH
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Binance Historical Trade Downloader")
    logger.info(f"Symbol     : {SYMBOL}")
    logger.info(f"Target     : {args.trades:,} trades")
    logger.info(f"Output     : {output_path}")
    logger.info("=" * 60)
    
    # Download
    raw_trades = download_agg_trades(
        target_trades=args.trades,
        hours_back=args.hours
    )
    
    if not raw_trades:
        logger.error("No trades downloaded!")
        sys.exit(1)
    
    # Convert to book ticks
    df = trades_to_book_ticks(raw_trades)
    
    if df.empty:
        logger.error("Conversion produced no ticks!")
        sys.exit(1)
    
    # Compute features
    df = compute_features(df)
    
    # Save
    df.to_csv(output_path, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"DOWNLOAD COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total rows  : {len(df):,}")
    logger.info(f"Date range  : {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    logger.info(f"Columns     : {list(df.columns)}")
    logger.info(f"File size   : {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
