"""
data_pipeline/schemas.py — Pydantic data models for market tick data.

MarketTick  : raw OHLCV from yfinance, serialised to/from JSON for Kafka.
ValidatedTick: tick after DataValidator processing with error annotations.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class MarketTick(BaseModel):
    """
    Raw OHLCV candle as fetched from yfinance.
    Serialised to JSON for Kafka transport; deserialised by consumer.
    """

    symbol: str = Field(..., description="Ticker symbol, e.g. 'AAPL'")
    timestamp: datetime = Field(..., description="UTC candle timestamp")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(..., ge=0.0, description="Trade volume")

    @field_validator("symbol")
    @classmethod
    def normalise_symbol(cls, v: str) -> str:
        """Strip whitespace and uppercase the ticker symbol."""
        return v.strip().upper()

    def to_json(self) -> str:
        """Serialise to JSON string for Kafka message value."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str | bytes) -> "MarketTick":
        """Deserialise from Kafka message bytes/string."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls.model_validate_json(raw)


class ValidatedTick(BaseModel):
    """
    MarketTick after DataValidator processing.

    is_valid=True  → tick passed all checks; safe for feature engineering.
    is_valid=False → at least one check failed; tick is quarantined.
    """

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_valid: bool = True
    validation_errors: List[str] = Field(default_factory=list)

    @classmethod
    def from_market_tick(
        cls,
        tick: MarketTick,
        errors: Optional[List[str]] = None,
    ) -> "ValidatedTick":
        """
        Construct a ValidatedTick from a raw MarketTick plus error list.

        Args:
            tick:   The original MarketTick.
            errors: Validation error messages; empty list means valid.
        """
        err = errors or []
        return cls(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            open=tick.open,
            high=tick.high,
            low=tick.low,
            close=tick.close,
            volume=tick.volume,
            is_valid=len(err) == 0,
            validation_errors=err,
        )

    def to_market_tick(self) -> MarketTick:
        """Downcast back to a MarketTick (drops validation metadata)."""
        return MarketTick(
            symbol=self.symbol,
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


if __name__ == "__main__":
    from datetime import timezone

    sample = MarketTick(
        symbol="aapl",
        timestamp=datetime.now(timezone.utc),
        open=182.5,
        high=185.0,
        low=181.0,
        close=184.0,
        volume=1_234_567.0,
    )
    print("✅ MarketTick created:")
    print(f"   symbol    = {sample.symbol}")
    print(f"   timestamp = {sample.timestamp.isoformat()}")

    raw_json = sample.to_json()
    print(f"   JSON      = {raw_json[:80]}...")

    recovered = MarketTick.from_json(raw_json)
    assert recovered.symbol == sample.symbol
    assert recovered.close == sample.close
    print("✅ Round-trip JSON serialisation OK")

    validated = ValidatedTick.from_market_tick(sample)
    print(f"✅ ValidatedTick: is_valid={validated.is_valid}")
