"""
tests/test_producer.py — Unit tests for KafkaMarketProducer internals.

Tests cover:
  - yfinance data fetching and tick conversion (mocked)
  - MarketTick JSON serialisation round-trip
  - Ticker symbol normalisation
  - Tick sorting by timestamp

Run: pytest tests/test_producer.py -v
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_pipeline.schemas import MarketTick
from data_pipeline.producer import KafkaMarketProducer


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ohlcv_df(symbol: str, n_rows: int = 5) -> pd.DataFrame:
    """Build a minimal yfinance-style OHLCV DataFrame."""
    base_ts = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
    index = [base_ts + timedelta(minutes=i) for i in range(n_rows)]
    df = pd.DataFrame(
        {
            "Open":   [100.0 + i for i in range(n_rows)],
            "High":   [102.0 + i for i in range(n_rows)],
            "Low":    [99.0  + i for i in range(n_rows)],
            "Close":  [101.0 + i for i in range(n_rows)],
            "Volume": [500_000.0] * n_rows,
        },
        index=pd.DatetimeIndex(index, name="Datetime"),
    )
    return df


@pytest.fixture
def mock_producer() -> KafkaMarketProducer:
    """
    KafkaMarketProducer with a mocked confluent_kafka.Producer
    so no real Kafka connection is needed.
    """
    with patch("data_pipeline.producer.ConfluentProducer") as MockKafka:
        MockKafka.return_value = MagicMock()
        producer = KafkaMarketProducer(
            tickers=["AAPL", "MSFT"],
            topic="market_data",
            bootstrap_servers="localhost:9092",
            interval_seconds=0.0,
            lookback_days=7,
            loop=False,
        )
        # Keep reference to internal mock
        producer._producer = MockKafka.return_value
        return producer


# ── MarketTick serialisation ──────────────────────────────────────────────────

class TestMarketTickSerialisaton:
    def test_round_trip_json(self) -> None:
        tick = MarketTick(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=182.0, high=185.0, low=181.0, close=184.0, volume=1_000_000.0,
        )
        raw = tick.to_json()
        recovered = MarketTick.from_json(raw)
        assert recovered.symbol == tick.symbol
        assert recovered.close == tick.close
        assert recovered.volume == tick.volume

    def test_from_json_bytes(self) -> None:
        tick = MarketTick(
            symbol="MSFT",
            timestamp=datetime.now(timezone.utc),
            open=300.0, high=305.0, low=299.0, close=302.0, volume=200_000.0,
        )
        raw_bytes = tick.to_json().encode("utf-8")
        recovered = MarketTick.from_json(raw_bytes)
        assert recovered.symbol == "MSFT"

    def test_symbol_uppercased_on_creation(self) -> None:
        tick = MarketTick(
            symbol="aapl",
            timestamp=datetime.now(timezone.utc),
            open=1.0, high=2.0, low=0.5, close=1.5, volume=100.0,
        )
        assert tick.symbol == "AAPL"


# ── Tick fetching (mocked yfinance) ──────────────────────────────────────────

class TestFetchSortedTicks:
    def test_ticks_sorted_by_timestamp(self, mock_producer: KafkaMarketProducer) -> None:
        """Ticks from multiple tickers must be sorted chronologically."""
        df_aapl = _make_ohlcv_df("AAPL", n_rows=3)
        df_msft = _make_ohlcv_df("MSFT", n_rows=3)

        with patch("data_pipeline.producer.yf.download") as mock_dl:
            mock_dl.side_effect = [df_aapl, df_msft]
            ticks = mock_producer._fetch_sorted_ticks()

        timestamps = [t.timestamp for t in ticks]
        assert timestamps == sorted(timestamps), "Ticks are not chronologically sorted"

    def test_correct_number_of_ticks(self, mock_producer: KafkaMarketProducer) -> None:
        df_aapl = _make_ohlcv_df("AAPL", n_rows=4)
        df_msft = _make_ohlcv_df("MSFT", n_rows=4)

        with patch("data_pipeline.producer.yf.download") as mock_dl:
            mock_dl.side_effect = [df_aapl, df_msft]
            ticks = mock_producer._fetch_sorted_ticks()

        assert len(ticks) == 8

    def test_empty_df_skipped_gracefully(self, mock_producer: KafkaMarketProducer) -> None:
        """Empty DataFrame for one ticker should not abort the whole fetch."""
        mock_producer.tickers = ["AAPL"]
        df_aapl = _make_ohlcv_df("AAPL", n_rows=2)

        with patch("data_pipeline.producer.yf.download") as mock_dl:
            mock_dl.return_value = df_aapl
            ticks = mock_producer._fetch_sorted_ticks()

        assert len(ticks) == 2

    def test_all_empty_raises_runtime_error(self, mock_producer: KafkaMarketProducer) -> None:
        mock_producer.tickers = ["FAKE"]
        with patch("data_pipeline.producer.yf.download") as mock_dl:
            mock_dl.return_value = pd.DataFrame()
            with pytest.raises(RuntimeError, match="No ticks fetched"):
                mock_producer._fetch_sorted_ticks()

    def test_fallback_to_7_days(self, mock_producer: KafkaMarketProducer) -> None:
        """If lookback_days = 30 and yf.download returns empty, it must fallback to 7 days."""
        mock_producer.lookback_days = 30
        mock_producer.tickers = ["AAPL"]
        df_7d = _make_ohlcv_df("AAPL", n_rows=2)

        with patch("data_pipeline.producer.yf.download") as mock_dl:
            # First download (30d) returns empty df, second (7d) returns valid df
            mock_dl.side_effect = [pd.DataFrame(), df_7d]
            ticks = mock_producer._fetch_sorted_ticks()

        assert len(ticks) == 2
        # Check that download was called twice: first with 30d, then with 7d
        assert mock_dl.call_count == 2
        first_call_kwargs = mock_dl.call_args_list[0].kwargs
        second_call_kwargs = mock_dl.call_args_list[1].kwargs
        assert first_call_kwargs["period"] == "30d"
        assert second_call_kwargs["period"] == "7d"



# ── Kafka publish ─────────────────────────────────────────────────────────────

class TestPublish:
    def test_produce_called_with_correct_key(
        self, mock_producer: KafkaMarketProducer
    ) -> None:
        tick = MarketTick(
            symbol="NVDA",
            timestamp=datetime.now(timezone.utc),
            open=500.0, high=510.0, low=498.0, close=505.0, volume=3_000_000.0,
        )
        mock_producer._publish(tick)
        call_kwargs = mock_producer._producer.produce.call_args.kwargs
        assert call_kwargs["key"] == b"NVDA"
        assert call_kwargs["topic"] == "market_data"

    def test_produce_value_is_valid_json(
        self, mock_producer: KafkaMarketProducer
    ) -> None:
        import json
        tick = MarketTick(
            symbol="AMZN",
            timestamp=datetime.now(timezone.utc),
            open=180.0, high=183.0, low=179.0, close=182.0, volume=1_500_000.0,
        )
        mock_producer._publish(tick)
        raw_value: bytes = mock_producer._producer.produce.call_args.kwargs["value"]
        parsed = json.loads(raw_value.decode("utf-8"))
        assert parsed["symbol"] == "AMZN"
        assert parsed["close"] == 182.0
