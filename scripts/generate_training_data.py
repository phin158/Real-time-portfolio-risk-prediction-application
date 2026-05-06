"""
scripts/generate_training_data.py

Fetches historical 1-minute OHLCV data from yfinance for training,
runs it through FeatureEngineer to generate the feature dataset,
and saves the result to a Parquet file.

Usage:
    python scripts/generate_training_data.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

# Adjust path so we can import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import get_settings
from data_pipeline.schemas import ValidatedTick
from feature_engineering.engineer import FeatureEngineer

logger = logging.getLogger(__name__)

def fetch_yfinance_history(tickers: list[str], period: str = "7d", interval: str = "1m") -> list[ValidatedTick]:
    """Fetch history from yfinance and convert directly to ValidatedTicks."""
    logger.info("Fetching %s of %s data for %d tickers...", period, interval, len(tickers))
    all_ticks = []
    
    for symbol in tickers:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty:
                logger.warning("No data for %s", symbol)
                continue
            
            # Handle MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            for ts, row in df.iterrows():
                # Ensure UTC
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    utc_ts = ts.to_pydatetime().astimezone(timezone.utc)
                else:
                    utc_ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)
                
                tick = ValidatedTick(
                    symbol=symbol,
                    timestamp=utc_ts,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                    is_valid=True
                )
                all_ticks.append(tick)
        except Exception as e:
            logger.error("Error fetching %s: %s", symbol, e)
    
    # Sort chronologically to simulate streaming
    all_ticks.sort(key=lambda t: t.timestamp)
    logger.info("Fetched %d total ticks.", len(all_ticks))
    return all_ticks

def generate_dataset(ticks: list[ValidatedTick], symbols: list[str]) -> pd.DataFrame:
    """Run ticks through FeatureEngineer and aggregate into a single DataFrame."""
    # We set a large history cap so we don't drop older rows during processing
    engineer = FeatureEngineer(symbols=symbols, history_cap=len(ticks) + 1000)
    
    logger.info("Processing ticks through FeatureEngineer...")
    for tick in ticks:
        engineer.update(tick)
    
    all_records = []
    for symbol in symbols:
        df = engineer.get_feature_df(symbol, lookback=engineer.history_cap)
        if not df.empty:
            # Reattach symbol
            df.reset_index(inplace=True)
            df.rename(columns={"index": "timestamp"}, inplace=True)
            df.insert(0, "symbol", symbol)
            all_records.append(df)
    
    if not all_records:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_records, ignore_index=True)
    combined_df.sort_values(by=["timestamp", "symbol"], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    
    return combined_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = get_settings()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=str, default="7d", help="yfinance period (e.g., 7d, 60d)")
    parser.add_argument("--output", type=str, default="data/raw/training_data.parquet")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    ticks = fetch_yfinance_history(cfg.tickers_list, period=args.period)
    if not ticks:
        logger.error("No data available to generate training set.")
        sys.exit(1)
        
    dataset_df = generate_dataset(ticks, cfg.tickers_list)
    if dataset_df.empty:
        logger.error("Dataset generation failed (maybe all data fell in warm-up period).")
        sys.exit(1)
        
    logger.info("Dataset shape: %s", dataset_df.shape)
    
    # Save to parquet
    dataset_df.to_parquet(args.output, index=False)
    logger.info("✅ Training data saved to %s", args.output)
