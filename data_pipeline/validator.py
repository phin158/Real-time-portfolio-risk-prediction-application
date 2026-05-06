"""
data_pipeline/validator.py — DataValidator class.

Validates raw MarketTick data for schema correctness, OHLC consistency,
outlier detection, and NaN/Inf values before entering feature engineering.
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

from data_pipeline.schemas import MarketTick, ValidatedTick

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Stateful validator for OHLCV market ticks.

    Maintains a per-symbol rolling last-close for outlier detection.
    Thread-safe for single-threaded consumers; use one instance per consumer.

    Args:
        max_price_multiplier: Reject tick if close deviates beyond this
            multiple of the previous close (default 10×).
        min_volume: Minimum acceptable trade volume (default 0.0).
    """

    def __init__(
        self,
        max_price_multiplier: float = 10.0,
        min_volume: float = 0.0,
    ) -> None:
        self.max_price_multiplier = max_price_multiplier
        self.min_volume = min_volume
        self._last_close: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────

    def validate(self, tick: MarketTick) -> ValidatedTick:
        """
        Run all validation checks on a single MarketTick.

        Args:
            tick: Raw tick from the Kafka consumer.

        Returns:
            ValidatedTick with is_valid=True if all checks pass,
            or is_valid=False with a populated validation_errors list.
        """
        errors: List[str] = []
        errors.extend(self._check_nan_inf(tick))
        errors.extend(self._check_prices_positive(tick))
        errors.extend(self._check_ohlc_consistency(tick))
        errors.extend(self._check_volume(tick))
        errors.extend(self._check_price_outlier(tick))

        validated = ValidatedTick.from_market_tick(tick, errors)

        if validated.is_valid:
            self._last_close[tick.symbol] = tick.close
            logger.debug("✅ %s tick valid — close=%.4f", tick.symbol, tick.close)
        else:
            logger.warning(
                "⚠️  %s tick INVALID — %s", tick.symbol, " | ".join(errors)
            )

        return validated

    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Clear the rolling close buffer.

        Args:
            symbol: If provided, reset only that symbol; otherwise reset all.
        """
        if symbol:
            self._last_close.pop(symbol, None)
        else:
            self._last_close.clear()

    # ── Private checks — each returns a (possibly empty) list of errors ───

    @staticmethod
    def _check_nan_inf(tick: MarketTick) -> List[str]:
        """Reject ticks containing NaN or Inf in numeric fields."""
        errors = []
        for name, value in [
            ("open", tick.open),
            ("high", tick.high),
            ("low", tick.low),
            ("close", tick.close),
            ("volume", tick.volume),
        ]:
            if not math.isfinite(value):
                errors.append(f"{name}={value} is not finite (NaN/Inf)")
        return errors

    @staticmethod
    def _check_prices_positive(tick: MarketTick) -> List[str]:
        """All OHLC prices must be strictly positive."""
        errors = []
        for name, value in [
            ("open", tick.open),
            ("high", tick.high),
            ("low", tick.low),
            ("close", tick.close),
        ]:
            if value <= 0:
                errors.append(f"{name}={value} must be > 0")
        return errors

    @staticmethod
    def _check_ohlc_consistency(tick: MarketTick) -> List[str]:
        """OHLC internal consistency checks."""
        errors = []
        if tick.high < tick.low:
            errors.append(f"high={tick.high} < low={tick.low}: impossible candle")
        if tick.high < tick.open:
            errors.append(f"high={tick.high} < open={tick.open}")
        if tick.high < tick.close:
            errors.append(f"high={tick.high} < close={tick.close}")
        if tick.low > tick.open:
            errors.append(f"low={tick.low} > open={tick.open}")
        if tick.low > tick.close:
            errors.append(f"low={tick.low} > close={tick.close}")
        return errors

    def _check_volume(self, tick: MarketTick) -> List[str]:
        """Volume must be at or above the configured minimum."""
        if tick.volume < self.min_volume:
            return [f"volume={tick.volume} < min_volume={self.min_volume}"]
        return []

    def _check_price_outlier(self, tick: MarketTick) -> List[str]:
        """
        Flag close price that deviates beyond max_price_multiplier from
        the previous close for the same symbol.
        """
        if tick.symbol not in self._last_close:
            return []  # No baseline yet — accept first tick unconditionally
        last = self._last_close[tick.symbol]
        if last == 0:
            return []
        ratio = tick.close / last
        threshold = self.max_price_multiplier
        if ratio > threshold or ratio < (1.0 / threshold):
            return [
                f"close={tick.close:.4f} deviates {ratio:.2f}× "
                f"from last_close={last:.4f} (max={threshold}×)"
            ]
        return []


if __name__ == "__main__":
    import sys
    from datetime import datetime, timezone

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    validator = DataValidator()

    # ── Valid tick ──────────────────────────────────────────────────────
    good = MarketTick(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        open=182.0, high=185.0, low=181.0, close=184.0, volume=500_000.0,
    )
    result = validator.validate(good)
    assert result.is_valid, f"Expected valid: {result.validation_errors}"
    print(f"✅ Valid tick accepted: {result.symbol} close={result.close}")

    # ── Invalid: high < low ─────────────────────────────────────────────
    bad = MarketTick(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        open=182.0, high=180.0, low=181.0, close=181.5, volume=100.0,
    )
    result2 = validator.validate(bad)
    assert not result2.is_valid
    print(f"✅ Invalid tick rejected: {result2.validation_errors}")

    # ── Outlier detection (close=2000 is >10× of last close=184) ───────
    # Ensure we have a baseline first (184.0 was set by the valid tick above)
    spike = MarketTick(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        open=2000.0, high=2005.0, low=1998.0, close=2000.0, volume=100.0,
    )
    result3 = validator.validate(spike)
    assert not result3.is_valid, f"Expected outlier rejection, got: {result3.validation_errors}"
    print(f"✅ Outlier tick rejected: {result3.validation_errors}")

    print("\n✅ All DataValidator checks passed.")
