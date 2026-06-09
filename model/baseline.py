"""
model/baseline.py — Statistical Baseline Risk Model.

Used when TFT checkpoint is unavailable (not trained yet, or file missing).
Provides statistically sound risk estimates WITHOUT neural network.

Methods:
- EWMA Volatility (lambda=0.94 per RiskMetrics):
    sigma^2_t = lambda * sigma^2_{t-1} + (1-lambda) * r^2_t
- Historical VaR (empirical quantile method)
- Historical CVaR (Expected Shortfall)

These are used as fallback to ensure the system NEVER outputs VaR from
random model weights (which would be meaningless).
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# EWMA decay factor (RiskMetrics standard for daily data)
# For 1-minute bars, the same lambda is commonly used
EWMA_LAMBDA: float = 0.94


class BaselineRiskModel:
    """
    Statistical baseline that computes per-symbol risk without a neural network.

    Interface is intentionally similar to RiskPredictor.predict() so the
    calling code can swap between TFT and baseline transparently.

    Args:
        ewma_lambda: Decay factor for EWMA volatility (default 0.94).
        ann_factor:  Annualisation multiplier for 1-minute bars.
    """

    def __init__(
        self,
        ewma_lambda: float = EWMA_LAMBDA,
        ann_factor: float = float(np.sqrt(252 * 390)),
    ) -> None:
        self.ewma_lambda = ewma_lambda
        self.ann_factor = ann_factor

    def predict(
        self,
        feature_tensor: np.ndarray,
        symbols: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-symbol risk using statistical methods only.

        Replaces TFT inference when checkpoint is missing.

        The feature_tensor layout (from FeatureEngineer):
            shape (lookback, n_symbols, n_features)
            feature[0] = log_return   ← used for historical VaR
            feature[1] = vol_30       ← used as proxy for EWMA vol

        Args:
            feature_tensor: shape (lookback, n_symbols, n_features)
            symbols:        list of N symbol names

        Returns:
            Dict mapping symbol -> {var_99, var_95, vol_forecast}
            (same keys as RiskPredictor.predict() output)
        """
        if feature_tensor.size == 0 or len(symbols) == 0:
            return {}

        lookback, n_symbols, n_features = feature_tensor.shape

        if n_symbols == 0 or lookback < 2:
            return {}

        # feature_tensor shape: (lookback, n_symbols, n_features)
        # Transpose to (n_symbols, lookback, n_features) for per-symbol access
        # feature index 0 = log_return
        log_returns = feature_tensor[:, :, 0]  # (lookback, n_symbols)

        results: Dict[str, Dict[str, float]] = {}

        for i, sym in enumerate(symbols):
            sym_returns = log_returns[:, i]  # shape (lookback,)

            # Filter out zero-padded warm-up entries
            non_zero_mask = sym_returns != 0.0
            if non_zero_mask.sum() < 5:
                results[sym] = {"var_99": 0.0, "var_95": 0.0, "vol_forecast": 0.0}
                continue

            real_returns = sym_returns[non_zero_mask]

            # ── EWMA Volatility ──────────────────────────────────────────────
            ewma_vol = self._ewma_volatility(real_returns, self.ewma_lambda)

            # ── Historical VaR ───────────────────────────────────────────────
            if len(real_returns) >= 20:
                q01 = float(np.quantile(real_returns, 0.01))
                q05 = float(np.quantile(real_returns, 0.05))
                var_99 = float(max(0.0, -q01))
                var_95 = float(max(0.0, -q05))
            else:
                # Fallback to parametric if too few observations
                var_99 = float(max(0.0, 2.3263 * ewma_vol))
                var_95 = float(max(0.0, 1.6449 * ewma_vol))

            results[sym] = {
                "var_99": var_99,
                "var_95": var_95,
                "vol_forecast": float(ewma_vol * self.ann_factor),
            }

        return results

    @staticmethod
    def _ewma_volatility(returns: np.ndarray, lam: float) -> float:
        """
        Compute EWMA variance and return annualised 1-step volatility.

        Formula (RiskMetrics):
            sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r^2_t

        Args:
            returns: 1-D array of log returns.
            lam:     Decay factor (0 < lambda < 1).

        Returns:
            Scalar volatility (not annualised — in same unit as returns).
        """
        if len(returns) == 0:
            return 0.0

        # Initialise with sample variance of first few observations
        var = float(np.var(returns[:min(10, len(returns))], ddof=0))
        if var == 0:
            var = 1e-8  # avoid zero initialisation

        for r in returns:
            var = lam * var + (1.0 - lam) * float(r) ** 2

        return float(np.sqrt(var))


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Testing BaselineRiskModel …")
    rng = np.random.default_rng(42)

    # Simulate feature tensor: (lookback=60, n_symbols=3, n_features=12)
    log_rets = rng.normal(0.0001, 0.01, size=(60, 3))
    other_features = rng.normal(0, 1, size=(60, 3, 11))  # 11 other features
    feature_tensor = np.concatenate(
        [log_rets[:, :, np.newaxis], other_features], axis=2
    )  # (60, 3, 12)

    model = BaselineRiskModel()
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = model.predict(feature_tensor, symbols)

    for sym, metrics in results.items():
        print(f"  {sym}: VaR99={metrics['var_99']:.4f}, VaR95={metrics['var_95']:.4f}, Vol={metrics['vol_forecast']:.4f}")
        assert metrics["var_99"] >= 0.0
        assert metrics["var_95"] >= 0.0
        assert metrics["vol_forecast"] >= 0.0
        assert metrics["var_99"] >= metrics["var_95"] or abs(metrics["var_99"] - metrics["var_95"]) < 1e-8

    print("\n✅ BaselineRiskModel self-test passed.")
    sys.exit(0)
