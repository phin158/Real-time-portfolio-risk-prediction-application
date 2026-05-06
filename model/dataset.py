"""
model/dataset.py — PyTorch Dataset for Time-Series forecasting.

Takes the parquet file produced by generate_training_data.py
and builds overlapping windows for TFT training.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class PortfolioRiskDataset(Dataset):
    """
    Dataset that returns (history_features, target_vars).
    
    History: `seq_len` timesteps of N_FEATURES for a single symbol.
    Target: The log_return (or VaR target) in the next `pred_len` timesteps.
    For simplicity, we target predicting the standard deviation (volatility) 
    over the next `pred_len` steps, as well as the immediate next log_return.
    
    Args:
        parquet_path: Path to the generated parquet file.
        seq_len: Number of historical steps per window.
        pred_len: Number of steps to forecast.
        symbols: Optional list of symbols to include.
    """
    
    def __init__(
        self,
        parquet_path: str,
        seq_len: int = 60,
        pred_len: int = 5,
        symbols: list[str] | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        logger.info("Loading dataset from %s", parquet_path)
        df = pd.read_parquet(parquet_path)
        
        if symbols:
            df = df[df["symbol"].isin(symbols)]
            
        self.samples = []
        
        feature_cols = [
            "log_return", "vol_30", "vol_60", "vol_390", 
            "rsi_14", "macd_line", "macd_signal", "macd_hist", "zscore_30"
        ]
        
        # Group by symbol and construct windows
        for sym, group in df.groupby("symbol"):
            group = group.sort_values("timestamp")
            features = group[feature_cols].values.astype(np.float32)
            
            # Normalize or fillna if necessary (FeatureEngineer already did nan_to_num but let's be safe)
            features = np.nan_to_num(features, nan=0.0)
            
            n_rows = len(features)
            for i in range(n_rows - seq_len - pred_len + 1):
                # Historical sequence
                x = features[i : i + seq_len]
                # Future sequence for target calculation
                y_future = features[i + seq_len : i + seq_len + pred_len]
                
                # Target: 
                # 1. Immediate next log return
                # 2. Standard deviation of log returns in the pred window (future volatility)
                future_returns = y_future[:, 0] # log_return is index 0
                target_ret = future_returns[0]
                target_vol = float(np.std(future_returns, ddof=1)) if pred_len > 1 else 0.0
                
                self.samples.append((x, np.array([target_ret, target_vol], dtype=np.float32)))
                
        logger.info("Constructed %d samples.", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        ds = PortfolioRiskDataset("data/raw/training_data.parquet", seq_len=60, pred_len=5)
        print(f"Dataset length: {len(ds)}")
        if len(ds) > 0:
            x, y = ds[0]
            print(f"X shape: {x.shape}, Y shape: {y.shape}")
    except FileNotFoundError:
        print("data/raw/training_data.parquet not found. Run generate_training_data.py first.")
