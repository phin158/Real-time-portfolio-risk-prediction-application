"""
tests/test_predictor.py

Unit tests for TFT model architecture and RiskPredictor inference.

Phase 2 change:
- test_risk_predictor_inference no longer expects meaningful predictions from missing checkpoint.
- The system must return {} (empty dict) when checkpoint is missing.
- This prevents random-weight predictions from being treated as real risk estimates.
"""
from __future__ import annotations

import torch
import numpy as np
import pytest

from model.tft import TemporalFusionTransformer, tft_loss, quantile_loss
from model.predictor import RiskPredictor
from model.baseline import BaselineRiskModel


# ── TFT Model Architecture Tests ─────────────────────────────────────────────

def test_tft_output_shape():
    """TFT must output (batch, 4) for [Q0.01, Q0.05, Q0.50, Vol]."""
    model = TemporalFusionTransformer(num_features=12, hidden_size=64)
    x = torch.randn(16, 60, 12)
    out = model(x)
    assert out.shape == (16, 4), f"Expected (16, 4), got {out.shape}"


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
    targets = torch.tensor([[1.0, 0.5]])  # Return=1.0, Vol=0.5
    # Vol loss = (0.5 - 0.5)^2 = 0
    # Total loss = quantile loss = 0.1866
    loss = tft_loss(preds, targets)
    assert loss.item() == pytest.approx(0.1866, abs=1e-3)


# ── RiskPredictor — Missing Checkpoint Tests ─────────────────────────────────

class TestMissingCheckpoint:
    """
    When checkpoint is missing, predict() must return {} (empty dict).
    The system must NEVER use random weights for VaR predictions.
    """

    def test_checkpoint_loaded_false_when_missing(self):
        """checkpoint_loaded must be False if file does not exist."""
        predictor = RiskPredictor(checkpoint_path="nonexistent_file.pt", device="cpu")
        assert predictor.checkpoint_loaded is False, (
            "checkpoint_loaded must be False when checkpoint file is missing"
        )

    def test_predict_returns_empty_dict_when_missing(self):
        """predict() must return {} when checkpoint is missing — NOT random predictions."""
        predictor = RiskPredictor(checkpoint_path="nonexistent_file.pt", device="cpu")
        feature_tensor = np.zeros((60, 2, 9), dtype=np.float32)
        symbols = ["AAPL", "MSFT"]

        results = predictor.predict(feature_tensor, symbols)

        assert results == {}, (
            "predict() must return empty dict when checkpoint is missing. "
            "Random-weight TFT predictions are meaningless and must not be returned."
        )

    def test_model_is_none_when_checkpoint_missing(self):
        """Model must not be initialised with random weights when checkpoint missing."""
        predictor = RiskPredictor(checkpoint_path="nonexistent_file.pt", device="cpu")
        assert predictor.model is None, (
            "Model must be None when checkpoint is missing to prevent random-weight inference"
        )

    def test_is_ready_false_when_missing(self):
        """is_ready property must be False when checkpoint is missing."""
        predictor = RiskPredictor(checkpoint_path="nonexistent_file.pt", device="cpu")
        assert predictor.is_ready is False


# ── RiskPredictor — NaN Guard Tests ──────────────────────────────────────────

class TestNaNGuard:
    """predict() must reject NaN/Inf tensors and return {} instead of crashing."""

    def _predictor_with_model(self) -> RiskPredictor:
        """Create a predictor with a model loaded (bypassing file check)."""
        predictor = RiskPredictor.__new__(RiskPredictor)
        predictor.device = torch.device("cpu")
        predictor.checkpoint_path = "fake.pt"
        predictor.num_features = 12
        predictor.model = TemporalFusionTransformer(num_features=12, hidden_size=64)
        predictor.checkpoint_loaded = True
        return predictor

    def test_nan_tensor_returns_empty(self):
        """NaN tensor must return {} (not garbage predictions)."""
        predictor = self._predictor_with_model()
        nan_tensor = np.full((60, 2, 12), np.nan, dtype=np.float32)
        result = predictor.predict(nan_tensor, ["AAPL", "MSFT"])
        assert result == {}, "NaN tensor must return empty dict"

    def test_inf_tensor_returns_empty(self):
        """Inf tensor must return {} (not garbage predictions)."""
        predictor = self._predictor_with_model()
        inf_tensor = np.full((60, 2, 12), np.inf, dtype=np.float32)
        result = predictor.predict(inf_tensor, ["AAPL", "MSFT"])
        assert result == {}, "Inf tensor must return empty dict"

    def test_valid_tensor_returns_results(self):
        """Valid (finite) tensor must return non-empty predictions."""
        predictor = self._predictor_with_model()
        valid_tensor = np.zeros((60, 2, 12), dtype=np.float32)
        result = predictor.predict(valid_tensor, ["AAPL", "MSFT"])

        assert len(result) == 2, "Valid tensor should return predictions for all symbols"
        for sym in ["AAPL", "MSFT"]:
            assert sym in result
            assert result[sym]["var_99"] >= 0.0
            assert result[sym]["var_95"] >= 0.0
            assert result[sym]["vol_forecast"] >= 0.0


# ── RiskPredictor — Edge Cases ────────────────────────────────────────────────

def test_risk_predictor_empty_input():
    """Empty tensor and empty symbol list must return {}."""
    predictor = RiskPredictor(checkpoint_path="nonexistent.pt", device="cpu")
    results = predictor.predict(np.zeros((0, 0, 12), dtype=np.float32), [])
    assert results == {}


# ── BaselineRiskModel Tests ───────────────────────────────────────────────────

class TestBaselineRiskModel:
    """Verify BaselineRiskModel produces valid fallback predictions."""

    def test_baseline_returns_predictions(self):
        """Baseline must return non-empty predictions for valid input."""
        rng = np.random.default_rng(42)
        log_rets = rng.normal(0.0001, 0.01, size=(60, 2))
        other = rng.normal(0, 1, size=(60, 2, 8))
        feature_tensor = np.concatenate(
            [log_rets[:, :, np.newaxis], other], axis=2
        )

        baseline = BaselineRiskModel()
        results = baseline.predict(feature_tensor, ["AAPL", "MSFT"])

        assert len(results) == 2
        for sym in ["AAPL", "MSFT"]:
            assert "var_99" in results[sym]
            assert "var_95" in results[sym]
            assert "vol_forecast" in results[sym]
            assert results[sym]["var_99"] >= 0.0
            assert results[sym]["var_95"] >= 0.0
            assert results[sym]["vol_forecast"] >= 0.0

    def test_baseline_empty_input(self):
        """Baseline must handle empty input without error."""
        baseline = BaselineRiskModel()
        result = baseline.predict(np.zeros((0, 0, 9), dtype=np.float32), [])
        assert result == {}

    def test_baseline_ewma_volatility_positive(self):
        """EWMA volatility must always be positive."""
        baseline = BaselineRiskModel()
        rng = np.random.default_rng(1)
        returns = rng.normal(0, 0.01, size=100)
        vol = baseline._ewma_volatility(returns, lam=0.94)
        assert vol > 0.0, "EWMA volatility must be positive"
