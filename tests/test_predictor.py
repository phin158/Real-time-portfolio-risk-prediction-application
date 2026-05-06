"""
tests/test_predictor.py

Unit tests for TFT model and RiskPredictor inference.
"""
from __future__ import annotations

import torch
import numpy as np
import pytest

from model.tft import TemporalFusionTransformer, tft_loss, quantile_loss
from model.predictor import RiskPredictor

def test_tft_output_shape():
    model = TemporalFusionTransformer(num_features=9, hidden_size=32)
    x = torch.randn(16, 60, 9)
    out = model(x)
    assert out.shape == (16, 4)

def test_quantile_loss():
    preds = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([1.0])
    # For q=0.5, err=1.0 -> loss = max(-0.5*1, 0.5*1) = 0.5
    # For q=0.01, err=1.0 -> loss = max(-0.99*1, 0.01*1) = 0.01
    # For q=0.05, err=1.0 -> loss = max(-0.95*1, 0.05*1) = 0.05
    # Mean of (0.01, 0.05, 0.5) = 0.56 / 3 = 0.1866
    ql = quantile_loss(preds, target, [0.01, 0.05, 0.5])
    assert ql.item() == pytest.approx(0.1866, abs=1e-3)

def test_tft_loss():
    preds = torch.tensor([[0.0, 0.0, 0.0, 0.5]])
    targets = torch.tensor([[1.0, 0.5]]) # Return=1.0, Vol=0.5
    # Vol loss = (0.5 - 0.5)^2 = 0
    # Total loss = quantile loss = 0.1866
    loss = tft_loss(preds, targets)
    assert loss.item() == pytest.approx(0.1866, abs=1e-3)

def test_risk_predictor_inference():
    # Will use initialized weights if no checkpoint exists
    predictor = RiskPredictor(checkpoint_path="nonexistent.pt", device="cpu")
    
    # feature tensor: (lookback, n_symbols, n_features)
    feature_tensor = np.zeros((60, 2, 9), dtype=np.float32)
    symbols = ["AAPL", "MSFT"]
    
    results = predictor.predict(feature_tensor, symbols)
    
    assert len(results) == 2
    assert "AAPL" in results
    assert "MSFT" in results
    
    for sym in symbols:
        assert "var_99" in results[sym]
        assert "var_95" in results[sym]
        assert "vol_forecast" in results[sym]
        
        # Check non-negative constraints
        assert results[sym]["var_99"] >= 0.0
        assert results[sym]["var_95"] >= 0.0
        assert results[sym]["vol_forecast"] >= 0.0

def test_risk_predictor_empty_input():
    predictor = RiskPredictor(checkpoint_path="nonexistent.pt", device="cpu")
    results = predictor.predict(np.zeros((0, 0, 9), dtype=np.float32), [])
    assert results == {}
