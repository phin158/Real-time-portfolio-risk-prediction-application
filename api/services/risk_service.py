"""
api/services/risk_service.py — Orchestrates FeatureEngineer + RiskPredictor + Redis.

Runs in the background (called by KafkaConsumer thread).
Computes features and predictions on every tick and publishes to Redis Pub/Sub.
"""
import json
import logging
import redis
from typing import Dict, Any

from config.settings import get_settings
from data_pipeline.schemas import ValidatedTick
from feature_engineering.engineer import FeatureEngineer
from model.predictor import RiskPredictor

logger = logging.getLogger(__name__)

class RiskService:
    def __init__(self):
        self.cfg = get_settings()
        self.feature_engineer = FeatureEngineer(symbols=self.cfg.tickers_list)
        self.predictor = RiskPredictor()
        
        # Use sync redis since this is called from a synchronous thread (Kafka polling)
        try:
            self.redis_client = redis.Redis.from_url(self.cfg.redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("Connected to Redis at %s", self.cfg.redis_url)
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            self.redis_client = None

    def process_tick(self, tick: ValidatedTick) -> None:
        """Called for every valid tick from Kafka."""
        record = self.feature_engineer.update(tick)
        
        # Wait until we have enough history to make predictions
        if record is None:
            return
            
        tensor = self.feature_engineer.get_portfolio_tensor(lookback=60)
        predictions = self.predictor.predict(tensor, self.feature_engineer.symbols)
        
        if predictions:
            payload = {
                "timestamp": tick.timestamp.isoformat(),
                "symbol_updated": tick.symbol,
                "current_price": tick.close,
                "predictions": predictions
            }
            
            if self.redis_client:
                try:
                    # Broadcast to WebSocket channel
                    self.redis_client.publish("portfolio_risk_stream", json.dumps(payload))
                except Exception as e:
                    logger.error("Redis publish error: %s", e)

    def get_current_metrics(self) -> Dict[str, Any]:
        """Called by REST API to get current snapshot."""
        tensor = self.feature_engineer.get_portfolio_tensor(lookback=60)
        predictions = self.predictor.predict(tensor, self.feature_engineer.symbols)
        
        corr = self.feature_engineer.get_correlation_matrix(window=60)
        # Convert index/columns to string and fill NaNs
        corr_dict = corr.fillna(0.0).to_dict() if not corr.empty else {}
        
        return {
            "predictions": predictions,
            "correlation_matrix": corr_dict,
            "ready_symbols": self.feature_engineer.get_ready_symbols(),
            "tick_counts": self.feature_engineer.tick_counts()
        }
