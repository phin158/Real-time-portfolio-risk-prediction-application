"""
model/predictor.py — RiskPredictor class.

Loads the trained TFT checkpoint ONCE at startup.
Provides inference methods for real-time predictions.
"""
from __future__ import annotations

import logging
import torch
import numpy as np
from typing import Dict, Tuple

from model.tft import TemporalFusionTransformer
from config.settings import get_settings

logger = logging.getLogger(__name__)

class RiskPredictor:
    """
    Wrapper for TFT model inference.
    Loads the weights on initialization.
    """
    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        cfg = get_settings()
        self.device = torch.device(device or cfg.device)
        self.checkpoint_path = checkpoint_path or cfg.model_checkpoint_path
        
        # Initialize model architecture
        self.model = TemporalFusionTransformer(num_features=9, hidden_size=32).to(self.device)
        self.model.eval()
        
        # Load weights if available
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded model weights from %s", self.checkpoint_path)
        except Exception as e:
            logger.warning("Could not load weights from %s: %s. Using initialized weights.", self.checkpoint_path, e)

    @torch.no_grad()
    def predict(self, feature_tensor: np.ndarray, symbols: list[str]) -> Dict[str, Dict[str, float]]:
        """
        Run inference on the feature tensor.
        
        Args:
            feature_tensor: shape (lookback, n_symbols, N_FEATURES)
            symbols: list of ticker symbols corresponding to n_symbols
            
        Returns:
            Dict mapping symbol -> {
                "var_99": float, 
                "var_95": float, 
                "vol_forecast": float
            }
        """
        # FeatureEngineer returns (lookback, n_symbols, features)
        # We need (n_symbols, lookback, features) for the model batch
        lookback, n_symbols, n_features = feature_tensor.shape
        if n_symbols == 0 or lookback == 0:
            return {}
            
        # Transpose to (n_symbols, lookback, n_features)
        x = np.transpose(feature_tensor, (1, 0, 2))
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Inference
        out = self.model(x_tensor) # (n_symbols, 4) -> [Q0.01, Q0.05, Q0.50, Vol]
        out_np = out.cpu().numpy()
        
        results = {}
        for i, sym in enumerate(symbols):
            q_01, q_05, q_50, vol = out_np[i]
            # VaR is typically expressed as a positive percentage loss
            # If Q0.01 is -0.05 (-5%), VaR99 is 5%
            results[sym] = {
                "var_99": float(max(0.0, -q_01)),
                "var_95": float(max(0.0, -q_05)),
                "vol_forecast": float(max(0.0, vol)) # Volatility must be positive
            }
            
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    predictor = RiskPredictor()
    dummy_tensor = np.random.randn(60, 2, 9).astype(np.float32)
    res = predictor.predict(dummy_tensor, ["AAPL", "MSFT"])
    print("Inference results:")
    for sym, metrics in res.items():
        print(f"  {sym}: {metrics}")
