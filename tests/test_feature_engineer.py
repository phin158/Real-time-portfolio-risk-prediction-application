"""
tests/test_feature_engineer.py — Unit tests for Phase 2.

Covers:
  - indicators.py: each function independently
  - FeatureEngineer: warm-up, feature shape, NaN handling,
    portfolio tensor, correlation matrix

Run: pytest tests/test_feature_engineer.py -v
"""
from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import List

import numpy as np
import pandas as pd
import pytest

from feature_engineering.indicators import (
    ANN_FACTOR,
    correlation_matrix,
    ema,
    log_return,
    macd,
    rolling_volatility,
    rolling_zscore,
    rsi,
)
from feature_engineering.engineer import (
    N_FEATURES,
    FEATURE_NAMES,
    FeatureEngineer,
    FeatureRecord,
)
from data_pipeline.schemas import ValidatedTick


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_prices(n: int = 100, seed: int = 0, base: float = 100.0) -> np.ndarray:
    """Generate synthetic close prices as a geometric random walk."""
    rng = np.random.default_rng(seed)
    log_rets = rng.normal(0.0, 0.001, n)
    return base * np.exp(np.cumsum(log_rets))


def make_validated_tick(
    symbol: str,
    close: float,
    ts: datetime,
) -> ValidatedTick:
    """Create a ValidatedTick with consistent OHLC from close."""
    return ValidatedTick(
        symbol=symbol,
        timestamp=ts,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=500_000.0,
        is_valid=True,
    )


def feed_ticks(
    engineer: FeatureEngineer,
    symbol: str,
    n: int,
    base_price: float = 100.0,
    seed: int = 0,
) -> List[FeatureRecord]:
    """Feed `n` synthetic ticks into engineer, return non-None records."""
    prices = make_prices(n, seed=seed, base=base_price)
    base_ts = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
    records = []
    for i, price in enumerate(prices):
        tick = make_validated_tick(symbol, price, base_ts + timedelta(minutes=i))
        rec = engineer.update(tick)
        if rec is not None:
            records.append(rec)
    return records


# ═════════════════════════════════════════════════════════════════════════════
# indicators.py
# ═════════════════════════════════════════════════════════════════════════════

class TestLogReturn:
    def test_length(self) -> None:
        prices = make_prices(50)
        rets = log_return(prices)
        assert len(rets) == len(prices) - 1

    def test_all_finite(self) -> None:
        prices = make_prices(50)
        rets = log_return(prices)
        assert np.all(np.isfinite(rets))

    def test_direction(self) -> None:
        """Rising price should give positive log return."""
        assert log_return(np.array([100.0, 110.0]))[0] > 0

    def test_too_short(self) -> None:
        assert len(log_return(np.array([100.0]))) == 0

    def test_zero_price_no_crash(self) -> None:
        """Zero prices must not crash; non-finite replaced with 0."""
        rets = log_return(np.array([100.0, 0.0, 100.0]))
        assert np.all(np.isfinite(rets))


class TestEMA:
    def test_length(self) -> None:
        prices = make_prices(50)
        assert len(ema(prices, span=12)) == len(prices)

    def test_single_value(self) -> None:
        out = ema(np.array([42.0]), span=5)
        np.testing.assert_allclose(out, [42.0])

    def test_trend(self) -> None:
        """EMA of monotonically rising series must be strictly increasing."""
        prices = np.arange(1, 51, dtype=float)
        e = ema(prices, span=5)
        assert np.all(np.diff(e) > 0)

    def test_empty(self) -> None:
        assert len(ema(np.empty(0), span=5)) == 0


class TestRollingVolatility:
    def test_finite_for_sufficient_data(self) -> None:
        rets = log_return(make_prices(200))
        vol = rolling_volatility(rets, window=30)
        assert np.isfinite(vol) and vol > 0

    def test_nan_for_insufficient_data(self) -> None:
        rets = log_return(make_prices(10))
        assert np.isnan(rolling_volatility(rets, window=30))

    def test_higher_vol_for_noisier_series(self) -> None:
        rng = np.random.default_rng(1)
        low_noise = rng.normal(0, 0.001, 200)
        high_noise = rng.normal(0, 0.01, 200)
        assert rolling_volatility(high_noise, window=60) > rolling_volatility(low_noise, window=60)

    def test_annualisation(self) -> None:
        """Daily std * ANN_FACTOR should scale correctly."""
        rets = np.full(60, 0.001)  # constant return
        vol = rolling_volatility(rets, window=60, ann_factor=ANN_FACTOR)
        assert vol == pytest.approx(0.0, abs=1e-6)  # zero std


class TestRSI:
    def test_range(self) -> None:
        prices = make_prices(200)
        val = rsi(prices)
        assert 0 <= val <= 100

    def test_nan_for_short_series(self) -> None:
        assert np.isnan(rsi(make_prices(10)))

    def test_constant_prices_returns_valid(self) -> None:
        """Constant prices → avg_loss = 0 → RSI = 100."""
        prices = np.full(50, 100.0)
        # All gains=0, all losses=0; RSI should return 100.0
        val = rsi(prices)
        assert val == pytest.approx(100.0)

    def test_uptrend_above_50(self) -> None:
        """Strongly rising series should yield RSI > 50."""
        prices = np.linspace(100, 200, 100)
        val = rsi(prices)
        assert val > 50


class TestMACD:
    def test_returns_three_values(self) -> None:
        prices = make_prices(200)
        result = macd(prices)
        assert len(result) == 3

    def test_finite_for_sufficient_data(self) -> None:
        prices = make_prices(200)
        ml, sl, hist = macd(prices)
        assert all(np.isfinite(v) for v in (ml, sl, hist))

    def test_nan_for_short_series(self) -> None:
        prices = make_prices(10)
        ml, sl, hist = macd(prices)
        assert all(np.isnan(v) for v in (ml, sl, hist))

    def test_hist_equals_line_minus_signal(self) -> None:
        prices = make_prices(200)
        ml, sl, hist = macd(prices)
        assert hist == pytest.approx(ml - sl, abs=1e-6)


class TestRollingZscore:
    def test_finite_for_sufficient_data(self) -> None:
        prices = make_prices(100)
        val = rolling_zscore(prices, window=30)
        assert np.isfinite(val)

    def test_nan_for_short_series(self) -> None:
        assert np.isnan(rolling_zscore(make_prices(10), window=30))

    def test_constant_prices_returns_zero(self) -> None:
        prices = np.full(50, 42.0)
        assert rolling_zscore(prices, window=30) == pytest.approx(0.0)

    def test_zscore_direction(self) -> None:
        """Price well above mean → positive z-score."""
        prices = np.concatenate([np.full(29, 100.0), [200.0]])
        assert rolling_zscore(prices, window=30) > 0


class TestCorrelationMatrix:
    def test_shape(self) -> None:
        rng = np.random.default_rng(5)
        d = {
            "AAPL": rng.normal(0, 0.01, 100),
            "MSFT": rng.normal(0, 0.01, 100),
        }
        corr = correlation_matrix(d, window=60)
        assert corr.shape == (2, 2)

    def test_diagonal_is_one(self) -> None:
        rng = np.random.default_rng(5)
        d = {"A": rng.normal(0, 0.01, 100), "B": rng.normal(0, 0.01, 100)}
        corr = correlation_matrix(d, window=60)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-6)

    def test_returns_empty_for_single_symbol(self) -> None:
        d = {"AAPL": np.random.default_rng(0).normal(0, 0.01, 100)}
        corr = correlation_matrix(d, window=60)
        assert corr.empty

    def test_insufficient_window_excluded(self) -> None:
        """Symbol with fewer than `window` bars should be excluded."""
        d = {
            "AAPL": np.random.default_rng(0).normal(0, 0.01, 100),
            "MSFT": np.random.default_rng(1).normal(0, 0.01, 10),  # too short
        }
        corr = correlation_matrix(d, window=60)
        assert corr.empty  # only 1 valid symbol → empty


# ═════════════════════════════════════════════════════════════════════════════
# FeatureEngineer
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer(symbols=["AAPL", "MSFT"], min_history=30)


class TestFeatureEngineerWarmup:
    def test_returns_none_before_min_history(self, engineer: FeatureEngineer) -> None:
        ts = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        for i in range(29):
            tick = make_validated_tick("AAPL", 100.0 + i * 0.01, ts + timedelta(minutes=i))
            result = engineer.update(tick)
            assert result is None, f"Expected None on tick {i+1}"

    def test_returns_record_at_min_history(self, engineer: FeatureEngineer) -> None:
        ts = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        result = None
        for i in range(30):
            tick = make_validated_tick("AAPL", 100.0 + i * 0.01, ts + timedelta(minutes=i))
            result = engineer.update(tick)
        assert result is not None

    def test_unknown_symbol_registered_dynamically(self, engineer: FeatureEngineer) -> None:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tick = make_validated_tick("NVDA", 500.0, ts)
        result = engineer.update(tick)
        assert "NVDA" in engineer.symbols
        assert result is None  # 1 tick < min_history


class TestFeatureRecord:
    def test_array_shape(self, engineer: FeatureEngineer) -> None:
        records = feed_ticks(engineer, "AAPL", n=100)
        assert len(records) > 0
        arr = records[-1].to_array()
        assert arr.shape == (N_FEATURES,)
        assert arr.dtype == np.float32

    def test_feature_names_match(self) -> None:
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_to_dict_keys(self, engineer: FeatureEngineer) -> None:
        records = feed_ticks(engineer, "AAPL", n=100)
        d = records[-1].to_dict()
        assert set(d.keys()) == set(FEATURE_NAMES)

    def test_rsi_in_valid_range(self, engineer: FeatureEngineer) -> None:
        records = feed_ticks(engineer, "AAPL", n=200)
        for rec in records[-50:]:
            if not np.isnan(rec.rsi_14):
                assert 0 <= rec.rsi_14 <= 100, f"RSI out of range: {rec.rsi_14}"

    def test_log_return_is_finite(self, engineer: FeatureEngineer) -> None:
        records = feed_ticks(engineer, "AAPL", n=100)
        for rec in records:
            assert np.isfinite(rec.log_return), f"log_return is not finite: {rec.log_return}"


class TestGetFeatureDF:
    def test_shape(self, engineer: FeatureEngineer) -> None:
        feed_ticks(engineer, "AAPL", n=200)
        df = engineer.get_feature_df("AAPL", lookback=60)
        assert df.shape == (60, N_FEATURES)

    def test_columns(self, engineer: FeatureEngineer) -> None:
        feed_ticks(engineer, "AAPL", n=100)
        df = engineer.get_feature_df("AAPL", lookback=10)
        assert list(df.columns) == FEATURE_NAMES

    def test_empty_for_unknown_symbol(self, engineer: FeatureEngineer) -> None:
        df = engineer.get_feature_df("ZZZZ")
        assert df.empty

    def test_partial_lookback(self, engineer: FeatureEngineer) -> None:
        """If fewer records than lookback, return all available."""
        feed_ticks(engineer, "AAPL", n=50)
        df = engineer.get_feature_df("AAPL", lookback=200)
        assert df.shape[0] <= 200
        assert df.shape[1] == N_FEATURES


class TestPortfolioTensor:
    def test_shape(self, engineer: FeatureEngineer) -> None:
        feed_ticks(engineer, "AAPL", n=200, seed=0)
        feed_ticks(engineer, "MSFT", n=200, seed=1)
        tensor = engineer.get_portfolio_tensor(lookback=60)
        assert tensor.shape == (60, 2, N_FEATURES)

    def test_dtype_float32(self, engineer: FeatureEngineer) -> None:
        feed_ticks(engineer, "AAPL", n=100, seed=0)
        feed_ticks(engineer, "MSFT", n=100, seed=1)
        tensor = engineer.get_portfolio_tensor(lookback=30)
        assert tensor.dtype == np.float32

    def test_no_nans_in_tensor(self, engineer: FeatureEngineer) -> None:
        """NaNs from warm-up rows must be replaced with 0.0."""
        feed_ticks(engineer, "AAPL", n=200, seed=0)
        feed_ticks(engineer, "MSFT", n=200, seed=1)
        tensor = engineer.get_portfolio_tensor(lookback=60)
        assert not np.any(np.isnan(tensor))

    def test_symbol_order(self, engineer: FeatureEngineer) -> None:
        """Symbol ordering must be consistent with engineer.symbols."""
        assert engineer.symbols[0] == "AAPL"
        assert engineer.symbols[1] == "MSFT"


class TestCorrelationMatrixIntegration:
    def test_shape_multi_symbol(self, engineer: FeatureEngineer) -> None:
        feed_ticks(engineer, "AAPL", n=200, seed=0)
        feed_ticks(engineer, "MSFT", n=200, seed=1)
        corr = engineer.get_correlation_matrix(window=60)
        assert corr.shape == (2, 2)

    def test_diagonal_ones(self, engineer: FeatureEngineer) -> None:
        feed_ticks(engineer, "AAPL", n=200, seed=0)
        feed_ticks(engineer, "MSFT", n=200, seed=1)
        corr = engineer.get_correlation_matrix(window=60)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-6)

    def test_symmetry(self, engineer: FeatureEngineer) -> None:
        feed_ticks(engineer, "AAPL", n=200, seed=0)
        feed_ticks(engineer, "MSFT", n=200, seed=1)
        corr = engineer.get_correlation_matrix(window=60)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)
