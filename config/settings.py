"""
config/settings.py — Centralised application configuration.

All values are read from environment variables or .env file.
Use get_settings() singleton throughout the application.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List
import os
import json

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from .env / environment variables.
    Immutable after initialisation — treat all attributes as read-only.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Kafka ─────────────────────────────────────────────────────────────
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Broker address(es), comma-separated",
    )
    kafka_topic_market_data: str = Field(
        default="market_data",
        description="Topic for raw OHLCV tick data",
    )
    kafka_group_id: str = Field(
        default="portfolio_consumer_group",
        description="Kafka consumer group ID",
    )
    kafka_security_protocol: str = Field(
        default="PLAINTEXT",
        description="Kafka security protocol",
    )

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_channel: str = Field(
        default="portfolio_risk_stream",
        description="Redis Pub/Sub channel for risk prediction broadcasts",
    )

    # ── API ───────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)

    # ── Data ──────────────────────────────────────────────────────────────
    tickers: str = Field(
        default="AAPL,MSFT,GOOGL,AMZN,NVDA",
        description="Comma-separated ticker symbols to stream",
    )
    stream_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Pause between streamed ticks (seconds)",
    )
    lookback_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description=(
            "Historical days of 1-min OHLCV to replay via yfinance. "
            "yfinance caps 1-minute data at ~30 days from today. "
            "If fetching 30 days fails (network, market-closed gaps), "
            "the producer logs a warning and uses whatever data is returned."
        ),
    )

    # ── Model ────────────────────────────────────────────────────────
    model_checkpoint_path: str = Field(
        default="model/checkpoints/tft_best.pt",
        description="Path to saved TFT checkpoint (.pt file)",
    )
    hidden_size: int = Field(
        default=64,
        ge=16,
        le=512,
        description=(
            "TFT hidden dimension. Default raised from 32 → 64 for better capacity. "
            "If a checkpoint trained with a different hidden_size is loaded, "
            "PyTorch will raise a shape error, predictor sets checkpoint_loaded=False, "
            "and the system falls back to BaselineRiskModel automatically."
        ),
    )
    device: str = Field(
        default="cpu",
        description="PyTorch compute device: cpu | cuda | mps",
    )

    # ── Training / Time-based Split ─────────────────────────────────────────
    # IMPORTANT: Replay/demo data MUST come after test_end_date.
    # The training pipeline enforces this separation to prevent data leakage.
    #
    # For yfinance 1-minute data (max ~30 days), the split may cover only
    # a few days in each period. Set to None to use all available data.
    train_end_date: str = Field(
        default="",
        description="Training data ends at this date (YYYY-MM-DD). Empty = use all data.",
    )
    val_end_date: str = Field(
        default="",
        description="Validation data ends at this date (YYYY-MM-DD).",
    )
    test_end_date: str = Field(
        default="",
        description="Test data ends at this date (YYYY-MM-DD).",
    )
    replay_start_date: str = Field(
        default="",
        description=(
            "Simulated real-time replay begins after this date. "
            "Must be after test_end_date to prevent data leakage."
        ),
    )

    # ── Loss Function Weights ───────────────────────────────────────────────
    # Quantile loss and volatility MSE have different scales —
    # 1:1 weighting (old) is unjustified. Default 0.1 for vol damping.
    lambda_quantile: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for quantile loss in combined TFT loss function.",
    )
    lambda_volatility: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight for volatility MSE loss in combined TFT loss function.",
    )

    # ── MLflow — Experiment Tracking (Phase 7) ─────────────────────────────
    # Defaults are safe to use even without MLflow server running —
    # connection is only established when mlflow.set_tracking_uri() is called.
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5001",
        description="MLflow tracking server URI. Port 5001 (not 5000) to avoid macOS AirPlay conflict.",
    )
    mlflow_experiment_name: str = Field(
        default="portfolio_risk_tft",
        description="MLflow experiment name to group all training runs under.",
    )

    # ── Admin Account (Phase 8) ───────────────────────────────────────────
    admin_username: str | None = Field(
        default=None,
        description="Default admin username. If set, created on startup if not exists.",
    )
    admin_password: str | None = Field(
        default=None,
        description="Default admin password.",
    )

    # ── Validators ────────────────────────────────────────────────────────
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Ensure device is a recognised PyTorch device string."""
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got '{v}'")
        return v

    # ── Derived properties ────────────────────────────────────────────────
    @property
    def tickers_list(self) -> List[str]:
        """Return tickers as a deduplicated, uppercased list. Reads from data/symbols.json if exists."""
        symbols_file = "data/symbols.json"
        if os.path.exists(symbols_file):
            try:
                with open(symbols_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        return list(dict.fromkeys(t.strip().upper() for t in data))
            except Exception as e:
                logger.error(f"Failed to read {symbols_file}: {e}")
                
        # Fallback to env / default
        return list(dict.fromkeys(t.strip().upper() for t in self.tickers.split(",")))

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        return f"redis://{self.redis_host}:{self.redis_port}"

    @property
    def kafka_producer_config(self) -> dict:
        """confluent-kafka Producer config dict."""
        return {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "security.protocol": self.kafka_security_protocol,
            "acks": "all",
            "retries": 3,
            "linger.ms": 5,
        }

    @property
    def kafka_consumer_config(self) -> dict:
        """confluent-kafka Consumer config dict."""
        return {
            "bootstrap.servers": self.kafka_bootstrap_servers,
            "security.protocol": self.kafka_security_protocol,
            "group.id": self.kafka_group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.
    Call this everywhere instead of constructing Settings() directly.
    """
    return Settings()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    print("✅ Settings loaded successfully")
    print(f"  kafka_bootstrap_servers : {s.kafka_bootstrap_servers}")
    print(f"  kafka_topic_market_data : {s.kafka_topic_market_data}")
    print(f"  tickers_list            : {s.tickers_list}")
    print(f"  redis_url               : {s.redis_url}")
    print(f"  stream_interval_seconds : {s.stream_interval_seconds}s")
    print(f"  device                  : {s.device}")
