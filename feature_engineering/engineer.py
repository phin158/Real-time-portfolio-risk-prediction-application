"""
feature_engineering/engineer.py — FeatureEngineer class.

Maintains a per-symbol rolling price buffer and computes a 9-feature
vector on every incoming ValidatedTick.  Outputs both per-symbol
DataFrames and a unified portfolio tensor for TFT model input.

Feature set (N_FEATURES = 9, per symbol per timestep):
  [0] log_return        — ln(close_t / close_{t-1})
  [1] vol_30            — annualised vol over last 30 bars
  [2] vol_60            — annualised vol over last 60 bars
  [3] vol_390           — annualised vol over last 390 bars (≈1 day)
  [4] rsi_14            — RSI with Wilder's smoothing
  [5] macd_line         — EMA(12) - EMA(26)
  [6] macd_signal       — EMA(9) of MACD line
  [7] macd_hist         — macd_line - macd_signal
  [8] zscore_30         — rolling z-score of close over 30 bars

Design:
  - Rolling deques are capped at MAX_WINDOW to bound memory.
  - Feature history deques are capped at HISTORY_CAP for tensor slicing.
  - MIN_HISTORY sets the warm-up period; features return None before it.
  - Thread-safety: not guaranteed — use one FeatureEngineer per consumer thread.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field, astuple
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_pipeline.schemas import ValidatedTick
from feature_engineering.indicators import (
    ANN_FACTOR,
    correlation_matrix,
    log_return,
    macd,
    rolling_volatility,
    rolling_zscore,
    rsi,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES: int = 9
FEATURE_NAMES: List[str] = [
    "log_return",
    "vol_30",
    "vol_60",
    "vol_390",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "zscore_30",
]
MIN_HISTORY: int = 30       # Minimum ticks before features are computed
MAX_WINDOW: int = 420       # Max price buffer per symbol (covers vol_390 + slack)
HISTORY_CAP: int = 1_000    # Max feature records kept per symbol


# ── FeatureRecord ─────────────────────────────────────────────────────────────

@dataclass
class FeatureRecord:
    """
    One row of computed features for a single symbol at a single timestamp.
    All values are float32 to minimise memory and match PyTorch default dtype.
    """
    timestamp: datetime
    symbol: str
    log_return: float
    vol_30: float
    vol_60: float
    vol_390: float
    rsi_14: float
    macd_line: float
    macd_signal: float
    macd_hist: float
    zscore_30: float

    def to_array(self) -> np.ndarray:
        """
        Return the 9 numeric features as a float32 numpy array.
        NaN values are preserved for downstream imputation.
        """
        return np.array(
            [
                self.log_return,
                self.vol_30,
                self.vol_60,
                self.vol_390,
                self.rsi_14,
                self.macd_line,
                self.macd_signal,
                self.macd_hist,
                self.zscore_30,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict:
        """Return feature as dict (excludes timestamp and symbol)."""
        return {k: v for k, v in zip(FEATURE_NAMES, self.to_array().tolist())}


# ── FeatureEngineer ───────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Stateful feature engine that consumes ValidatedTick objects and
    maintains per-symbol rolling buffers and feature histories.

    Args:
        symbols:     List of expected ticker symbols.
        min_history: Number of ticks required before features are emitted.
        max_window:  Maximum price buffer length per symbol.
        history_cap: Maximum feature records retained per symbol.
        ann_factor:  Annualisation factor for volatility (default: sqrt(252*390)).

    Usage:
        engineer = FeatureEngineer(['AAPL', 'MSFT'])
        record = engineer.update(validated_tick)   # returns None until warm-up
        tensor = engineer.get_portfolio_tensor(lookback=60)
    """

    def __init__(
        self,
        symbols: List[str],
        min_history: int = MIN_HISTORY,
        max_window: int = MAX_WINDOW,
        history_cap: int = HISTORY_CAP,
        ann_factor: float = ANN_FACTOR,
    ) -> None:
        self.symbols = [s.upper() for s in symbols]
        self.min_history = min_history
        self.max_window = max_window
        self.history_cap = history_cap
        self.ann_factor = ann_factor

        # Per-symbol rolling price buffers
        self._prices: Dict[str, deque[float]] = {
            s: deque(maxlen=max_window) for s in self.symbols
        }
        # Per-symbol feature history
        self._history: Dict[str, deque[FeatureRecord]] = {
            s: deque(maxlen=history_cap) for s in self.symbols
        }
        # Count of ticks processed per symbol (for logging)
        self._tick_count: Dict[str, int] = {s: 0 for s in self.symbols}

        logger.info(
            "FeatureEngineer initialised — symbols=%s  min_history=%d",
            self.symbols,
            self.min_history,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, tick: ValidatedTick) -> Optional[FeatureRecord]:
        """
        Ingest a new validated tick and compute features if warmed up.

        Args:
            tick: Validated OHLCV tick from DataValidator.

        Returns:
            FeatureRecord if enough history, else None during warm-up.
        """
        symbol = tick.symbol
        if symbol not in self._prices:
            logger.debug("Unknown symbol %s — registering dynamically", symbol)
            self._prices[symbol] = deque(maxlen=self.max_window)
            self._history[symbol] = deque(maxlen=self.history_cap)
            self._tick_count[symbol] = 0
            if symbol not in self.symbols:
                self.symbols.append(symbol)

        self._prices[symbol].append(tick.close)
        self._tick_count[symbol] += 1

        prices_arr = np.array(self._prices[symbol], dtype=np.float64)
        if len(prices_arr) < self.min_history:
            return None

        record = self._compute_features(symbol, tick.timestamp, prices_arr)
        self._history[symbol].append(record)

        logger.debug(
            "Feature computed: %s  lr=%.5f  vol30=%.4f  rsi=%.1f",
            symbol,
            record.log_return,
            record.vol_30,
            record.rsi_14,
        )
        return record

    def get_feature_df(self, symbol: str, lookback: int = 60) -> pd.DataFrame:
        """
        Return the last `lookback` feature records for a symbol as a DataFrame.

        Args:
            symbol:   Ticker symbol (case-insensitive).
            lookback: Number of most-recent records to return.

        Returns:
            DataFrame with columns = FEATURE_NAMES, index = timestamps.
            Empty DataFrame if no history available.
        """
        sym = symbol.upper()
        if sym not in self._history or len(self._history[sym]) == 0:
            logger.warning("No feature history for %s", sym)
            return pd.DataFrame(columns=FEATURE_NAMES)

        records = list(self._history[sym])[-lookback:]
        rows = [rec.to_dict() for rec in records]
        idx = [rec.timestamp for rec in records]
        return pd.DataFrame(rows, index=idx, columns=FEATURE_NAMES)

    def get_portfolio_tensor(self, lookback: int = 60) -> np.ndarray:
        """
        Build a (lookback, n_symbols, N_FEATURES) float32 tensor.

        Symbols are ordered as self.symbols.  If a symbol has fewer than
        `lookback` records, its rows are zero-padded on the left.

        Args:
            lookback: Time dimension of the output tensor.

        Returns:
            np.ndarray of shape (lookback, n_symbols, N_FEATURES), dtype float32.
        """
        n = len(self.symbols)
        tensor = np.zeros((lookback, n, N_FEATURES), dtype=np.float32)

        for col_idx, sym in enumerate(self.symbols):
            if sym not in self._history:
                continue
            records = list(self._history[sym])[-lookback:]
            n_available = len(records)
            if n_available == 0:
                continue

            # Right-align: fill the last n_available rows
            start_row = lookback - n_available
            for row_idx, rec in enumerate(records):
                arr = rec.to_array()
                # Replace NaN with 0.0 for model input
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                tensor[start_row + row_idx, col_idx, :] = arr

        return tensor

    def get_correlation_matrix(self, window: int = 60) -> pd.DataFrame:
        """
        Compute the rolling correlation matrix across all symbols.

        Args:
            window: Number of log-return bars to include.

        Returns:
            Pearson correlation DataFrame (n_symbols × n_symbols),
            or empty DataFrame if fewer than 2 symbols are ready.
        """
        returns_dict: Dict[str, np.ndarray] = {}
        for sym in self.symbols:
            prices_arr = np.array(self._prices.get(sym, []), dtype=np.float64)
            if len(prices_arr) >= 2:
                returns_dict[sym] = log_return(prices_arr)

        return correlation_matrix(returns_dict, window=window)

    def get_ready_symbols(self) -> List[str]:
        """Return symbols that have completed their warm-up period."""
        return [
            s for s in self.symbols
            if len(self._history.get(s, [])) > 0
        ]

    def tick_counts(self) -> Dict[str, int]:
        """Return number of ticks processed per symbol."""
        return dict(self._tick_count)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_features(
        self,
        symbol: str,
        timestamp: datetime,
        prices_arr: np.ndarray,
    ) -> FeatureRecord:
        """
        Compute all 9 features from a price buffer array.

        Args:
            symbol:     Ticker symbol.
            timestamp:  Timestamp of the latest bar.
            prices_arr: Array of close prices, latest last.

        Returns:
            FeatureRecord with all computed feature values.
        """
        rets = log_return(prices_arr)

        lr = float(rets[-1]) if len(rets) > 0 else np.nan
        v30 = rolling_volatility(rets, window=30, ann_factor=self.ann_factor)
        v60 = rolling_volatility(rets, window=60, ann_factor=self.ann_factor)
        v390 = rolling_volatility(rets, window=390, ann_factor=self.ann_factor)
        rsi_val = rsi(prices_arr)
        ml, sl, mh = macd(prices_arr)
        zs = rolling_zscore(prices_arr, window=30)

        return FeatureRecord(
            timestamp=timestamp,
            symbol=symbol,
            log_return=lr,
            vol_30=v30,
            vol_60=v60,
            vol_390=v390,
            rsi_14=rsi_val,
            macd_line=ml,
            macd_signal=sl,
            macd_hist=mh,
            zscore_30=zs,
        )


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from datetime import timezone, timedelta

    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    print("Testing FeatureEngineer …")

    rng = np.random.default_rng(0)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    engineer = FeatureEngineer(symbols=symbols, min_history=30)

    # ── Simulate 450 ticks per symbol ───────────────────────────────────────
    n_ticks = 450
    base_prices = {"AAPL": 180.0, "MSFT": 370.0, "GOOGL": 155.0}
    base_ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    last_records = {}

    for i in range(n_ticks):
        ts = base_ts + timedelta(minutes=i)
        for sym in symbols:
            price = base_prices[sym] * np.exp(rng.normal(0, 0.001))
            base_prices[sym] = price
            tick = ValidatedTick(
                symbol=sym,
                timestamp=ts,
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=rng.integers(100_000, 2_000_000),
                is_valid=True,
            )
            record = engineer.update(tick)
            if record is not None:
                last_records[sym] = record

    print(f"  ✅ Processed {n_ticks} ticks × {len(symbols)} symbols")

    # ── Verify all symbols are ready ─────────────────────────────────────────
    ready = engineer.get_ready_symbols()
    assert set(ready) == set(symbols), f"Not all symbols ready: {ready}"
    print(f"  ✅ All symbols warm-up complete: {ready}")

    # ── Check last FeatureRecord ──────────────────────────────────────────────
    rec = last_records["AAPL"]
    assert rec.symbol == "AAPL"
    assert np.isfinite(rec.log_return), f"log_return is not finite: {rec.log_return}"
    assert np.isfinite(rec.vol_30),     f"vol_30 is not finite: {rec.vol_30}"
    assert 0 <= rec.rsi_14 <= 100,      f"rsi_14 out of range: {rec.rsi_14}"
    arr = rec.to_array()
    assert arr.shape == (N_FEATURES,),  f"Feature array shape: {arr.shape}"
    assert arr.dtype == np.float32,     f"Feature dtype: {arr.dtype}"
    print(f"  ✅ FeatureRecord — shape={arr.shape}  dtype={arr.dtype}")
    print(f"     log_return={rec.log_return:.5f}  vol_30={rec.vol_30:.4f}  rsi_14={rec.rsi_14:.2f}")

    # ── get_feature_df ────────────────────────────────────────────────────────
    df = engineer.get_feature_df("AAPL", lookback=60)
    assert df.shape == (60, N_FEATURES), f"DataFrame shape: {df.shape}"
    assert list(df.columns) == FEATURE_NAMES, f"Columns mismatch: {df.columns.tolist()}"
    assert df.index.dtype != "object" or hasattr(df.index[0], "hour"), "Index is not datetime"
    print(f"  ✅ get_feature_df  — shape={df.shape}  columns={df.columns.tolist()}")

    # ── get_portfolio_tensor ──────────────────────────────────────────────────
    tensor = engineer.get_portfolio_tensor(lookback=60)
    assert tensor.shape == (60, len(symbols), N_FEATURES), f"Tensor shape: {tensor.shape}"
    assert tensor.dtype == np.float32, f"Tensor dtype: {tensor.dtype}"
    assert not np.any(np.isnan(tensor)), "Tensor contains NaN (should be 0-filled)"
    print(f"  ✅ get_portfolio_tensor — shape={tensor.shape}  dtype={tensor.dtype}")

    # ── get_correlation_matrix ────────────────────────────────────────────────
    corr = engineer.get_correlation_matrix(window=60)
    assert corr.shape == (len(symbols), len(symbols)), f"Corr shape: {corr.shape}"
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-6)
    print(f"  ✅ get_correlation_matrix — shape={corr.shape}")
    print(corr.round(3))

    print(f"\n✅ FeatureEngineer self-test PASSED — tensor shape={tensor.shape}")
    sys.exit(0)
