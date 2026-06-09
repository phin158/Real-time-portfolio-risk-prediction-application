"""
model/predictor.py — RiskPredictor class.

Loads the trained TFT checkpoint ONCE at startup.
Provides inference methods for real-time predictions.

IMPORTANT CHANGE (Phase 2):
If no checkpoint is found, this class sets checkpoint_loaded=False and
predict() returns an empty dict. The calling code (RiskService) is
responsible for switching to the statistical baseline.

The system MUST NOT produce AI VaR predictions from random weights.
Random-weight predictions are meaningless and would give users false confidence.
"""
from __future__ import annotations

import logging
import os
import numpy as np
import torch
from typing import Dict, List, Optional

from model.tft import TemporalFusionTransformer
from config.settings import get_settings

# ── MLflow (Phase 7) — optional, default use_registry=False — must NOT break tests
try:
    import mlflow.pytorch as _mlflow_pytorch
    _MLFLOW_AVAILABLE = True
except ImportError:
    _mlflow_pytorch = None  # type: ignore
    _MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)

# Total feature count: 12 after Phase 5 (9 original + 3 volume features)
_NUM_FEATURES_DEFAULT = 12  # volume_change, volume_zscore_30, dollar_volume added


class CheckpointNotFoundError(RuntimeError):
    """Raised when TFT checkpoint file does not exist."""


class RiskPredictor:
    """
    Wrapper for TFT model inference.

    Attributes:
        checkpoint_loaded (bool): True only if a valid checkpoint was loaded.
                                  If False, predict() returns {} and a warning
                                  is logged. RiskService then uses BaselineRiskModel.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        num_features: int = _NUM_FEATURES_DEFAULT,
        use_registry: bool = False,  # MUST default to False — True requires MLflow server
    ) -> None:
        cfg = get_settings()
        self.device = torch.device(device or cfg.device)
        self.checkpoint_path = checkpoint_path or cfg.model_checkpoint_path
        self.num_features = num_features
        self.hidden_size = cfg.hidden_size  # default 64; read from settings/env

        # ── Check checkpoint exists BEFORE building model ──────────────────
        self.checkpoint_loaded: bool = False

        # ── Option A: Load from MLflow Registry (only when use_registry=True) ──
        # This is entirely optional and never used by default. Existing tests
        # always use use_registry=False (the default) and are unaffected.
        if use_registry and _MLFLOW_AVAILABLE:
            try:
                self.model = _mlflow_pytorch.load_model(
                    "models:/PortfolioRiskTFT/Production"
                )
                self.checkpoint_loaded = True
                logger.info("Loaded TFT model from MLflow Registry (Production stage)")
                return  # done — skip file-based loading below
            except Exception as e:
                logger.warning(
                    "MLflow Registry load failed (%s). "
                    "Falling back to checkpoint file at '%s'.",
                    e, self.checkpoint_path,
                )
                # Fall through to file-based loading below

        if not os.path.isfile(self.checkpoint_path):
            logger.warning(
                "TFT checkpoint not found at '%s'. "
                "predict() will return empty results. "
                "RiskService will use statistical baseline instead.",
                self.checkpoint_path,
            )
            # Do NOT initialise the model with random weights —
            # that would produce meaningless VaR predictions.
            self.model: Optional[TemporalFusionTransformer] = None
            return

        # ── Build model and load weights ─────────────────────────────────
        # NOTE: hidden_size comes from settings (default 64).
        # If the checkpoint was trained with a different hidden_size,
        # load_state_dict() will raise a RuntimeError (shape mismatch),
        # which is caught below → checkpoint_loaded=False → baseline fallback.
        self.model = TemporalFusionTransformer(
            num_features=self.num_features,
            hidden_size=self.hidden_size,
        ).to(self.device)
        self.model.eval()

        try:
            state_dict = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(state_dict)
            self.checkpoint_loaded = True
            logger.info(
                "✅ TFT checkpoint loaded from '%s'", self.checkpoint_path
            )
        except Exception as e:
            logger.error(
                "Failed to load checkpoint '%s': %s. "
                "Disabling TFT inference. Statistical baseline will be used.",
                self.checkpoint_path,
                e,
            )
            self.model = None
            self.checkpoint_loaded = False

    @torch.no_grad()
    def predict(
        self,
        feature_tensor: np.ndarray,
        symbols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Run TFT inference on the feature tensor.

        Returns empty dict {} if checkpoint was not loaded, signalling
        RiskService to use the statistical baseline instead.

        Args:
            feature_tensor: shape (lookback, n_symbols, N_FEATURES)
            symbols:        list of ticker symbols corresponding to n_symbols

        Returns:
            Dict mapping symbol -> {var_99, var_95, vol_forecast}
            OR {} if checkpoint not loaded (use baseline instead).
        """
        # Guard: no checkpoint → no inference
        if not self.checkpoint_loaded or self.model is None:
            return {}

        lookback, n_symbols, n_features = feature_tensor.shape
        if n_symbols == 0 or lookback == 0:
            return {}

        # ── NaN / Inf guard ──────────────────────────────────────────────────
        if not np.isfinite(feature_tensor).all():
            nan_count = np.sum(~np.isfinite(feature_tensor))
            logger.warning(
                "Feature tensor contains %d NaN/Inf values — skipping TFT inference.",
                nan_count,
            )
            return {}

        # Transpose to (n_symbols, lookback, n_features) for batch inference
        x = np.transpose(feature_tensor, (1, 0, 2))
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)

        # Inference: output shape (n_symbols, 4) → [Q0.01, Q0.05, Q0.50, Vol]
        out = self.model(x_tensor)
        out_np = out.cpu().numpy()

        results: Dict[str, Dict[str, float]] = {}
        for i, sym in enumerate(symbols):
            q_01, q_05, q_50, vol = out_np[i]

            # Volatility is un-annualized 1-min standard deviation from TFT.
            # We use abs() because the linear layer might output small negative values,
            # and we annualize it to match the baseline model scale.
            results[sym] = {
                "var_99": float(max(0.0, -q_01)),
                "var_95": float(max(0.0, -q_05)),
                "vol_forecast": float(abs(vol) * np.sqrt(252 * 390)),
            }

        return results

    @property
    def is_ready(self) -> bool:
        """True if a valid checkpoint was loaded and the model is ready."""
        return self.checkpoint_loaded and self.model is not None


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    # Test 1: Missing checkpoint
    predictor = RiskPredictor(checkpoint_path="nonexistent_checkpoint.pt", device="cpu")
    assert not predictor.checkpoint_loaded, "checkpoint_loaded must be False if file missing"
    result = predictor.predict(np.zeros((60, 2, 12), dtype=np.float32), ["AAPL", "MSFT"])
    assert result == {}, "predict() must return {} when checkpoint missing"
    print("✅ Test 1 passed: missing checkpoint returns empty dict (not random weights)")

    # Test 2: NaN guard
    predictor2 = RiskPredictor(checkpoint_path="nonexistent_checkpoint.pt", device="cpu")
    # Manually set loaded=True to test NaN guard in isolation
    predictor2.checkpoint_loaded = True
    from model.tft import TemporalFusionTransformer
    predictor2.model = TemporalFusionTransformer(num_features=12, hidden_size=32)
    nan_tensor = np.full((60, 2, 12), np.nan, dtype=np.float32)
    result_nan = predictor2.predict(nan_tensor, ["AAPL", "MSFT"])
    assert result_nan == {}, "predict() must return {} for NaN tensor"
    print("✅ Test 2 passed: NaN tensor returns empty dict")

    print("\n✅ All predictor self-tests passed.")
    sys.exit(0)
