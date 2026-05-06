"""
config/settings.py — Centralised application configuration.

All values are read from environment variables or .env file.
Use get_settings() singleton throughout the application.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

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
        default="risk_updates",
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
        default=7,
        ge=1,
        le=365,
        description="Historical days of 1-min OHLCV to replay",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_checkpoint_path: str = Field(
        default="model/checkpoints/tft_best.pt",
        description="Path to saved TFT checkpoint (.pt file)",
    )
    device: str = Field(
        default="cpu",
        description="PyTorch compute device: cpu | cuda | mps",
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
        """Return tickers as a deduplicated, uppercased list."""
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
