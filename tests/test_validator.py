"""
tests/test_validator.py — Unit tests for DataValidator.

Run: pytest tests/test_validator.py -v
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from data_pipeline.schemas import MarketTick, ValidatedTick
from data_pipeline.validator import DataValidator


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_tick(**overrides) -> MarketTick:
    """Return a valid MarketTick, with optional field overrides.

    OHLC is auto-derived from close so consistency checks always pass
    unless high/low/open are explicitly overridden in `overrides`.
    """
    close = overrides.get("close", 184.0)
    defaults = dict(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        open=close,
        high=close * 1.005,   # +0.5%
        low=close * 0.995,    # -0.5%
        close=close,
        volume=500_000.0,
    )
    defaults.update(overrides)
    return MarketTick(**defaults)


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator(max_price_multiplier=10.0, min_volume=0.0)


# ── Happy-path ────────────────────────────────────────────────────────────────

class TestValidTick:
    def test_valid_tick_passes(self, validator: DataValidator) -> None:
        tick = make_tick()
        result = validator.validate(tick)
        assert result.is_valid
        assert result.validation_errors == []

    def test_valid_tick_updates_last_close(self, validator: DataValidator) -> None:
        tick = make_tick(close=184.0)
        validator.validate(tick)
        assert validator._last_close["AAPL"] == 184.0

    def test_first_tick_always_accepted(self, validator: DataValidator) -> None:
        """First tick for a symbol should pass outlier check regardless of value."""
        tick = make_tick(symbol="NEW", close=99999.0)
        result = validator.validate(tick)
        assert result.is_valid


# ── NaN / Inf ─────────────────────────────────────────────────────────────────

class TestNanInf:
    def test_nan_close_rejected(self, validator: DataValidator) -> None:
        tick = make_tick(close=float("nan"), high=float("nan"))
        result = validator.validate(tick)
        assert not result.is_valid
        assert any("not finite" in e for e in result.validation_errors)

    def test_inf_volume_rejected(self, validator: DataValidator) -> None:
        tick = make_tick(volume=float("inf"))
        result = validator.validate(tick)
        assert not result.is_valid


# ── Positive prices ───────────────────────────────────────────────────────────

class TestPositivePrices:
    def test_zero_close_rejected(self, validator: DataValidator) -> None:
        tick = make_tick(close=0.0, low=0.0)
        result = validator.validate(tick)
        assert not result.is_valid

    def test_negative_open_rejected(self, validator: DataValidator) -> None:
        tick = make_tick(open=-1.0, low=-1.0)
        result = validator.validate(tick)
        assert not result.is_valid


# ── OHLC consistency ──────────────────────────────────────────────────────────

class TestOHLCConsistency:
    def test_high_less_than_low_rejected(self, validator: DataValidator) -> None:
        tick = make_tick(high=180.0, low=181.0, open=180.5, close=180.5)
        result = validator.validate(tick)
        assert not result.is_valid
        assert any("impossible candle" in e for e in result.validation_errors)

    def test_high_less_than_close_rejected(self, validator: DataValidator) -> None:
        tick = make_tick(high=183.0, close=184.0)
        result = validator.validate(tick)
        assert not result.is_valid

    def test_low_greater_than_open_rejected(self, validator: DataValidator) -> None:
        tick = make_tick(low=183.0, open=182.0)
        result = validator.validate(tick)
        assert not result.is_valid


# ── Volume ────────────────────────────────────────────────────────────────────

class TestVolume:
    def test_zero_volume_accepted_by_default(self, validator: DataValidator) -> None:
        tick = make_tick(volume=0.0)
        result = validator.validate(tick)
        assert result.is_valid

    def test_volume_below_min_rejected(self) -> None:
        v = DataValidator(min_volume=100.0)
        tick = make_tick(volume=50.0)
        result = v.validate(tick)
        assert not result.is_valid


# ── Outlier detection ─────────────────────────────────────────────────────────

class TestOutlierDetection:
    def test_10x_spike_rejected(self, validator: DataValidator) -> None:
        # Seed the baseline
        validator.validate(make_tick(close=100.0))
        spike = make_tick(close=1001.0, open=1001.0, high=1001.0, low=1000.0)
        result = validator.validate(spike)
        assert not result.is_valid
        assert any("deviates" in e for e in result.validation_errors)

    def test_normal_move_accepted(self, validator: DataValidator) -> None:
        validator.validate(make_tick(close=100.0))
        normal = make_tick(close=101.0)
        result = validator.validate(normal)
        assert result.is_valid


# ── Reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_all_clears_buffer(self, validator: DataValidator) -> None:
        validator.validate(make_tick(symbol="AAPL", close=100.0))
        validator.validate(make_tick(symbol="MSFT", close=200.0))
        validator.reset()
        assert validator._last_close == {}

    def test_reset_symbol_clears_only_that_symbol(self, validator: DataValidator) -> None:
        validator.validate(make_tick(symbol="AAPL", close=100.0))
        validator.validate(make_tick(symbol="MSFT", close=200.0))
        validator.reset("AAPL")
        assert "AAPL" not in validator._last_close
        assert "MSFT" in validator._last_close
