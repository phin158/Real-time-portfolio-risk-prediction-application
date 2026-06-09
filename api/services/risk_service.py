"""
api/services/risk_service.py — Orchestrates FeatureEngineer + RiskPredictor + Redis.

Runs in the background (called by KafkaConsumer thread).
Computes features and predictions on every tick and publishes to Redis Pub/Sub.

Phase 2 change: Added checkpoint fallback logic.
If TFT checkpoint is not found, the system switches to BaselineRiskModel
(EWMA + Historical VaR) and clearly marks predictions as statistical baseline.
The system NEVER uses random-weight TFT predictions.
"""
import json
import logging
import redis
from typing import Dict, Any, List, Optional

import numpy as np

from config.settings import get_settings
from data_pipeline.schemas import ValidatedTick
from feature_engineering.engineer import FeatureEngineer
from model.predictor import RiskPredictor
from model.baseline import BaselineRiskModel
from portfolio.risk_aggregator import PortfolioRiskAggregator

logger = logging.getLogger(__name__)

# Covariance window for portfolio risk (needs 390 bars for stable estimation)
COVARIANCE_WINDOW = 390

# Minimum bars needed before any inference
MIN_HISTORY_BASIC = 60
MIN_HISTORY_FULL = 390


class RiskService:
    def __init__(self):
        self.cfg = get_settings()
        self.feature_engineer = FeatureEngineer(symbols=self.cfg.tickers_list)
        self.predictor = RiskPredictor()
        self.baseline = BaselineRiskModel()
        self.portfolio_aggregator = PortfolioRiskAggregator()

        # Determine which prediction mode to use
        if self.predictor.checkpoint_loaded:
            logger.info("✅ TFT checkpoint loaded — using TFT model for per-symbol predictions.")
            self._prediction_method = "tft"
        else:
            logger.warning(
                "⚠️  TFT checkpoint NOT loaded. "
                "Using statistical baseline (EWMA + Historical VaR). "
                "To use TFT, run: python scripts/train_tft.py"
            )
            self._prediction_method = "statistical_baseline"

        # Use sync redis since this is called from a synchronous thread (Kafka polling)
        try:
            self.redis_client = redis.Redis.from_url(self.cfg.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis at %s", self.cfg.redis_url)
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            self.redis_client = None

    def _get_per_symbol_predictions(
        self,
        feature_tensor: np.ndarray,
        symbols: List[str],
    ) -> tuple[Dict, str, Optional[str]]:
        """
        Get per-symbol risk predictions, automatically falling back to baseline
        if TFT checkpoint is not loaded.

        Returns:
            (predictions_dict, method_used, warning_message)
        """
        if self.predictor.checkpoint_loaded:
            # Try TFT first
            preds = self.predictor.predict(feature_tensor, symbols)
            if preds:  # Non-empty result from TFT
                return preds, "tft", None
            # TFT returned empty (e.g. NaN guard triggered) → fall back
            logger.warning("TFT returned empty result (possible NaN). Falling back to baseline.")

        # Baseline fallback
        if feature_tensor.size > 0:
            preds = self.baseline.predict(feature_tensor, symbols)
        else:
            preds = {}

        warning = (
            "TFT checkpoint not found. Using baseline risk model (EWMA + Historical VaR). "
            "To use TFT, run: python scripts/train_tft.py"
        )
        return preds, "statistical_baseline", warning

    def process_tick(self, tick: ValidatedTick) -> None:
        """Called for every valid tick from Kafka."""
        record = self.feature_engineer.update(tick)

        # Wait until we have enough history to make predictions
        if record is None:
            return

        tensor = self.feature_engineer.get_portfolio_tensor(lookback=60)
        predictions, method, warning = self._get_per_symbol_predictions(
            tensor, self.feature_engineer.symbols
        )

        if predictions:
            payload = {
                "timestamp": tick.timestamp.isoformat(),
                "symbol_updated": tick.symbol,
                "current_price": tick.close,
                "predictions": predictions,
                "method_used": method,
            }
            if warning:
                payload["warning"] = warning

            if self.redis_client:
                try:
                    # Broadcast to WebSocket channel — use cfg.redis_channel (not hardcoded)
                    self.redis_client.publish(self.cfg.redis_channel, json.dumps(payload))
                except Exception as e:
                    logger.error("Redis publish error: %s", e)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Called by REST API to get current snapshot."""
        tensor = self.feature_engineer.get_portfolio_tensor(lookback=60)
        predictions, method, warning = self._get_per_symbol_predictions(
            tensor, self.feature_engineer.symbols
        )

        corr = self.feature_engineer.get_correlation_matrix(window=60)
        corr_dict = corr.fillna(0.0).to_dict() if not corr.empty else {}

        result: Dict[str, Any] = {
            "predictions": predictions,
            "correlation_matrix": corr_dict,
            "ready_symbols": self.feature_engineer.get_ready_symbols(),
            "tick_counts": self.feature_engineer.tick_counts(),
            "method_used": method,
        }
        if warning:
            result["warning"] = warning

        return result

    def get_portfolio_risk(
        self,
        weights: Dict[str, float],
        covariance_window: int = COVARIANCE_WINDOW,
    ) -> Dict[str, Any]:
        """Compute portfolio-level risk using the CORRECT covariance-based method.

        This replaces the old weighted-average VaR (sum(w_i * var_i)) which
        was mathematically wrong because it ignored asset correlations.

        New approach:
          1. Get aligned log-return matrix from FeatureEngineer
          2. Compute covariance matrix from actual return data
          3. Use PortfolioRiskAggregator to compute:
             - Parametric VaR: -(mu_p + z * sigma_p), where sigma_p = sqrt(w.T @ Sigma @ w)
             - Historical VaR: empirical quantiles of weighted portfolio returns
             - CVaR: expected shortfall beyond VaR threshold calculated via historical simulation
                     on the weighted portfolio returns distribution.
          4. Return full risk report with reliability flags and method_used

        Args:
            weights:           Dict mapping symbol -> weight (will be normalised).
            covariance_window: Number of recent bars to use for covariance estimation.

        Returns:
            Dict with portfolio VaR, CVaR, volatility, matrices, and metadata.
        """
        # Only include symbols that are requested AND have data
        requested_symbols = [s.upper() for s in weights.keys()]
        available = self.feature_engineer.get_ready_symbols()
        included_symbols = [s for s in requested_symbols if s in available]

        if not included_symbols:
            return {
                "weights_used": {},
                "n_observations": 0,
                "portfolio_return_latest": 0.0,
                "portfolio_volatility_horizon": 0.0,
                "portfolio_volatility_annualized": 0.0,
                "portfolio_var_95_parametric": 0.0,
                "portfolio_var_99_parametric": 0.0,
                "portfolio_var_95_historical": 0.0,
                "portfolio_var_99_historical": 0.0,
                "portfolio_cvar_95": 0.0,
                "portfolio_cvar_99": 0.0,
                "correlation_matrix": {},
                "covariance_matrix": {},
                "method_used": "none",
                "reliable": False,
                "warning": "No symbols have sufficient history yet. Still warming up.",
                "per_symbol_predictions": {},
            }

        # Get aligned returns matrix (T, N) for included symbols
        returns_matrix, actual_symbols = self.feature_engineer.get_returns_matrix(
            symbols=included_symbols,
            window=covariance_window,
        )



        if returns_matrix.size == 0 or len(actual_symbols) == 0:
            return {
                "weights_used": weights,
                "n_observations": 0,
                "portfolio_return_latest": 0.0,
                "portfolio_volatility_horizon": 0.0,
                "portfolio_volatility_annualized": 0.0,
                "portfolio_var_95_parametric": 0.0,
                "portfolio_var_99_parametric": 0.0,
                "portfolio_var_95_historical": 0.0,
                "portfolio_var_99_historical": 0.0,
                "portfolio_cvar_95": 0.0,
                "portfolio_cvar_99": 0.0,
                "correlation_matrix": {},
                "covariance_matrix": {},
                "method_used": "none",
                "reliable": False,
                "warning": (
                    "Insufficient return history for any symbol. "
                    f"Minimum {MIN_HISTORY_BASIC} bars required. Still warming up."
                ),
                "per_symbol_predictions": {},
            }

        # Build weight vector aligned to actual_symbols order
        weight_values = np.array(
            [weights.get(s, 0.0) for s in actual_symbols],
            dtype=np.float64,
        )

        # Log excluded symbols
        excluded = [s for s in requested_symbols if s not in actual_symbols]
        if excluded:
            logger.warning(
                "Symbols excluded from portfolio risk (insufficient history): %s", excluded
            )

        # Compute portfolio risk using the correct covariance-based approach
        result = self.portfolio_aggregator.compute_portfolio_risk(
            returns_matrix=returns_matrix,
            weights=weight_values,
            symbols=actual_symbols,
        )

        # Attach info about excluded symbols
        if excluded:
            existing_warning = result.get("warning") or ""
            excluded_msg = f"Symbols excluded (no history): {excluded}."
            result["warning"] = (
                f"{existing_warning} {excluded_msg}".strip()
                if existing_warning
                else excluded_msg
            )
            result["reliable"] = False

        # Also include per-symbol predictions (TFT or baseline)
        tensor = self.feature_engineer.get_portfolio_tensor(lookback=60)
        per_symbol_preds, pred_method, pred_warning = self._get_per_symbol_predictions(
            tensor, self.feature_engineer.symbols
        )
        result["per_symbol_predictions"] = per_symbol_preds
        result["prediction_method"] = pred_method

        # Note in warning if using baseline
        if pred_warning:
            existing = result.get("warning") or ""
            result["warning"] = (
                f"{existing} {pred_warning}".strip() if existing else pred_warning
            )

        # Portfolio-level method remains covariance-based regardless of TFT availability
        result["method_used"] = "parametric+historical"

        return result
