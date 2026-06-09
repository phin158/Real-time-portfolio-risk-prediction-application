"""
feature_engineering/indicators.py — Pure-function technical indicator library.

All functions are stateless and operate on numpy arrays.
They are designed to be fast, vectorised where possible, and called
from FeatureEngineer with pre-sliced buffers.

Annualisation convention (1-minute bars):
  - 1 trading day  = 390 minutes
  - 1 trading year = 252 days
  - ANN_FACTOR = sqrt(252 * 390) ≈ 313.47
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────

ANN_FACTOR: float = float(np.sqrt(252 * 390))   # ≈ 313.47 for 1-min bars
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9


# ── Log Returns ───────────────────────────────────────────────────────────────

def log_return(prices: np.ndarray) -> np.ndarray:
    """
    Compute simple log returns for a price series.

    Returns array of length len(prices)-1.
    Non-finite values (zero prices, NaN) are replaced with 0.0.

    Args:
        prices: 1-D array of close prices, chronologically ordered.

    Returns:
        Array of log returns: ln(p_t / p_{t-1}).
    """
    if len(prices) < 2:
        return np.empty(0, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(prices[1:] / prices[:-1])

    return np.where(np.isfinite(ret), ret, 0.0)


# ── Exponential Moving Average ────────────────────────────────────────────────

def ema(values: np.ndarray, span: int) -> np.ndarray:
    """
    Exponential moving average (Wilder / pandas-equivalent decay).

    Uses alpha = 2 / (span + 1).  First value initialised to values[0].

    Args:
        values: 1-D numeric array.
        span:   EMA span (number of periods).

    Returns:
        EMA array, same length as `values`.
    """
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)

    alpha = 2.0 / (span + 1)
    out = np.empty(len(values), dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


# ── Rolling Volatility ────────────────────────────────────────────────────────

def rolling_volatility(
    log_returns: np.ndarray,
    window: int,
    ann_factor: float = ANN_FACTOR,
) -> float:
    """
    Annualised rolling volatility = std(log_returns[-window:]) * ann_factor.

    Args:
        log_returns: Array of log returns (all available history).
        window:      Look-back window in bars.
        ann_factor:  Annualisation multiplier (default: sqrt(252*390)).

    Returns:
        Scalar annualised volatility, or np.nan if insufficient data.
    """
    if len(log_returns) < max(window, 2):
        return np.nan

    subset = log_returns[-window:]
    std = float(np.std(subset, ddof=1))
    return std * ann_factor


# ── RSI ───────────────────────────────────────────────────────────────────────

def rsi(prices: np.ndarray, period: int = RSI_PERIOD) -> float:
    """
    Relative Strength Index using Wilder's smoothing.

    Args:
        prices: 1-D price array, at least (period + 1) elements long.
        period: RSI look-back period (default 14).

    Returns:
        RSI scalar [0, 100], or np.nan if insufficient data.
    """
    if len(prices) < period + 1:
        return np.nan

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed with simple average of first `period` bars
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Wilder's smoothing for remaining bars
    for g, l in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


# ── MACD ──────────────────────────────────────────────────────────────────────

def macd(
    prices: np.ndarray,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> Tuple[float, float, float]:
    """
    Compute MACD line, signal line, and histogram for the last bar.

    Args:
        prices: 1-D price array, at least `slow` elements.
        fast:   Fast EMA span (default 12).
        slow:   Slow EMA span (default 26).
        signal: Signal EMA span (default 9).

    Returns:
        Tuple (macd_line, signal_line, histogram), all np.nan if insufficient data.
    """
    nan_triple = (np.nan, np.nan, np.nan)
    if len(prices) < slow:
        return nan_triple

    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line_arr = ema_fast - ema_slow

    if len(macd_line_arr) < signal:
        return nan_triple

    signal_arr = ema(macd_line_arr, signal)
    ml = float(macd_line_arr[-1])
    sl = float(signal_arr[-1])
    return ml, sl, ml - sl


# ── Rolling Z-Score ───────────────────────────────────────────────────────────

def rolling_zscore(prices: np.ndarray, window: int) -> float:
    """
    Z-score of the last price relative to a rolling mean/std.

    Args:
        prices: 1-D price array.
        window: Look-back window for mean/std calculation.

    Returns:
        Scalar z-score, or np.nan if insufficient data or zero std.
    """
    if len(prices) < window:
        return np.nan

    subset = prices[-window:]
    mean = float(np.mean(subset))
    std = float(np.std(subset, ddof=1))

    if std == 0.0:
        return 0.0

    return float((prices[-1] - mean) / std)


# ── Correlation Matrix ────────────────────────────────────────────────────────

def correlation_matrix(
    returns_dict: Dict[str, np.ndarray],
    window: int,
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix from per-symbol log-return arrays.

    Only symbols with at least `window` returns are included.

    Args:
        returns_dict: Mapping symbol → 1-D array of log returns.
        window:       Number of most-recent returns to use.

    Returns:
        DataFrame (n_symbols × n_symbols) with correlation values,
        or empty DataFrame if fewer than 2 symbols have enough data.
    """
    data: Dict[str, np.ndarray] = {}
    for symbol, rets in returns_dict.items():
        if len(rets) >= window:
            data[symbol] = rets[-window:]

    if len(data) < 2:
        return pd.DataFrame()

    return pd.DataFrame(data).corr(method="pearson")


# ── Volume-Based Features ─────────────────────────────────────────────────────

def volume_change(volumes: np.ndarray) -> float:
    """
    Compute rate of change of the most recent volume bar relative to
    the previous bar.

    Formula:
        volume_change = (V_t - V_{t-1}) / V_{t-1}

    Args:
        volumes: 1-D array of volume values, at least length 2.

    Returns:
        Relative volume change in [-inf, +inf], or 0.0 if insufficient data.
    """
    if len(volumes) < 2:
        return 0.0
    v_prev = float(volumes[-2])
    v_curr = float(volumes[-1])
    if v_prev == 0.0:
        return 0.0
    change = (v_curr - v_prev) / v_prev
    return float(change) if np.isfinite(change) else 0.0


def volume_zscore(volumes: np.ndarray, window: int = 30) -> float:
    """
    Compute rolling z-score of volume over the last `window` bars.

    Formula:
        mu  = mean(volumes[-window:])
        std = std(volumes[-window:])
        zscore = (V_t - mu) / std

    A high positive z-score indicates unusually high volume (possible breakout).
    A low negative z-score indicates unusually low volume.

    Args:
        volumes: 1-D array of volume values.
        window:  Look-back window (default 30).

    Returns:
        Z-score of the latest volume bar, or 0.0 if insufficient data.
    """
    if len(volumes) < window:
        return 0.0
    window_vols = volumes[-window:]
    mu  = float(np.mean(window_vols))
    std = float(np.std(window_vols, ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    z = (float(volumes[-1]) - mu) / std
    return float(z) if np.isfinite(z) else 0.0


def dollar_volume(closes: np.ndarray, volumes: np.ndarray) -> float:
    """
    Compute rolling dollar volume (close × volume) for the latest bar.

    Dollar volume is a proxy for market liquidity and institutional activity.
    High dollar volume with price movement typically signals stronger conviction.

    Formula:
        dollar_volume_t = close_t × volume_t

    Args:
        closes:  1-D array of close prices (same length as volumes).
        volumes: 1-D array of trade volumes.

    Returns:
        Dollar volume for the latest bar, or 0.0 if insufficient data.
    """
    if len(closes) == 0 or len(volumes) == 0:
        return 0.0
    dv = float(closes[-1]) * float(volumes[-1])
    return float(dv) if np.isfinite(dv) else 0.0


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("Testing indicators.py …")
    rng = np.random.default_rng(42)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0, 0.001, 500))

    # log_return
    rets = log_return(prices)
    assert len(rets) == len(prices) - 1, "log_return length mismatch"
    assert np.all(np.isfinite(rets)), "log_return contains non-finite values"
    print(f"  ✅ log_return  — shape={rets.shape}  mean={rets.mean():.6f}")

    # rolling_volatility
    vol_30 = rolling_volatility(rets, window=30)
    vol_60 = rolling_volatility(rets, window=60)
    assert np.isfinite(vol_30), "vol_30 is nan/inf"
    assert np.isfinite(vol_60), "vol_60 is nan/inf"
    print(f"  ✅ rolling_vol — vol_30={vol_30:.4f}  vol_60={vol_60:.4f}")

    # rsi
    rsi_val = rsi(prices)
    assert 0 <= rsi_val <= 100, f"RSI out of range: {rsi_val}"
    print(f"  ✅ rsi         — RSI(14)={rsi_val:.2f}")

    # macd
    ml, sl, hist = macd(prices)
    assert all(np.isfinite(v) for v in (ml, sl, hist)), "MACD contains nan/inf"
    print(f"  ✅ macd        — line={ml:.4f}  signal={sl:.4f}  hist={hist:.4f}")

    # rolling_zscore
    zs = rolling_zscore(prices, window=30)
    assert np.isfinite(zs), f"zscore is not finite: {zs}"
    print(f"  ✅ zscore      — z={zs:.4f}")

    # correlation_matrix
    syms = {"AAPL": rets, "MSFT": rets + rng.normal(0, 0.0005, len(rets))}
    corr = correlation_matrix(syms, window=60)
    assert corr.shape == (2, 2), f"Correlation matrix shape: {corr.shape}"
    print(f"  ✅ corr_matrix — shape={corr.shape}  AAPL-MSFT={corr.loc['AAPL','MSFT']:.4f}")

    # insufficient data → nan
    assert np.isnan(rolling_volatility(rets[:5], window=30)), "Expected nan for short series"
    assert np.isnan(rsi(prices[:5])), "Expected nan for short series"
    ml2, sl2, h2 = macd(prices[:10])
    assert all(np.isnan(v) for v in (ml2, sl2, h2)), "Expected nan for short series"
    print("  ✅ NaN guard   — all short-series checks pass")

    # volume indicators
    vols_arr = rng.uniform(1e5, 5e5, 50)
    cl_arr   = prices[1:51]
    vc = volume_change(vols_arr)
    assert np.isfinite(vc), f"volume_change not finite: {vc}"
    vz = volume_zscore(vols_arr, window=30)
    assert np.isfinite(vz), f"volume_zscore not finite: {vz}"
    dv = dollar_volume(cl_arr, vols_arr)
    assert dv > 0, f"dollar_volume must be positive: {dv}"
    print(f"  ✅ volume_change={vc:.4f}  volume_zscore={vz:.4f}  dollar_volume={dv:.0f}")

    print("\n✅ All indicators.py tests passed.")
    sys.exit(0)

