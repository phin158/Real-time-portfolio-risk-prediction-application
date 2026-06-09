"""
data_pipeline/producer.py — KafkaMarketProducer

Fetches historical 1-minute OHLCV data from yfinance for a list of tickers,
then replays it tick-by-tick to a Kafka topic at a configurable rate,
simulating a real-time market data feed.

Design decisions:
  - Uses confluent-kafka (librdkafka) for high-throughput, low-latency delivery.
  - Replays data in chronological order, interleaving tickers by timestamp.
  - Loops replay indefinitely (loop=True) for continuous demo operation.
  - Delivery callbacks log per-message success/failure without blocking.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional

import pandas as pd
import yfinance as yf
from confluent_kafka import Producer as ConfluentProducer
from confluent_kafka import KafkaError

from config.settings import get_settings
from data_pipeline.schemas import MarketTick

logger = logging.getLogger(__name__)


class KafkaMarketProducer:
    """
    Streams mock-realtime OHLCV ticks to Kafka by replaying yfinance history.

    Args:
        tickers:           List of ticker symbols to stream (e.g. ['AAPL', 'MSFT']).
        topic:             Kafka topic name to publish to.
        bootstrap_servers: Kafka broker address.
        interval_seconds:  Delay between published ticks (seconds).
        lookback_days:     Days of 1-minute OHLCV history to replay.
        loop:              If True, replay indefinitely; otherwise replay once.
    """

    def __init__(
        self,
        tickers: List[str],
        topic: str,
        bootstrap_servers: str,
        interval_seconds: float = 1.0,
        lookback_days: int = 7,
        loop: bool = True,
    ) -> None:
        self.tickers = tickers
        self.topic = topic
        self.interval_seconds = interval_seconds
        self.lookback_days = lookback_days
        self.loop = loop

        self._producer = ConfluentProducer(
            {
                "bootstrap.servers": bootstrap_servers,
                "acks": "all",
                "retries": 3,
                "linger.ms": 5,
                "compression.type": "snappy",
            }
        )
        logger.info(
            "KafkaMarketProducer initialised — brokers=%s topic=%s tickers=%s",
            bootstrap_servers,
            topic,
            tickers,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the streaming loop.

        Fetches yfinance data once (or reuses cached data on repeat loops),
        then publishes ticks with the configured interval.
        Blocks indefinitely when loop=True.
        """
        logger.info("📡 Fetching yfinance data (lookback=%dd)…", self.lookback_days)
        ticks = self._fetch_sorted_ticks()
        logger.info("📦 Loaded %d ticks — starting stream", len(ticks))

        iteration = 0
        while True:
            iteration += 1
            logger.info("🔁 Replay iteration %d (%d ticks)", iteration, len(ticks))
            for tick in ticks:
                self._publish(tick)
                time.sleep(self.interval_seconds)
            self._producer.flush()
            if not self.loop:
                break
            logger.info("♻️  Loop complete — restarting replay")

    def close(self) -> None:
        """Flush pending messages and close the producer."""
        self._producer.flush()
        logger.info("KafkaMarketProducer closed — all messages flushed")

    # ── Private helpers ───────────────────────────────────────────────────

    def _fetch_sorted_ticks(self) -> List[MarketTick]:
        """
        Download 1-minute OHLCV for all tickers and return a flat list
        of MarketTick objects sorted ascending by timestamp.
        """
        lookback_to_try = self.lookback_days
        all_ticks = self._download_ticks_for_period(lookback_to_try)

        if not all_ticks and lookback_to_try > 7:
            logger.warning(
                "⚠️  Failed to fetch %d days of 1-minute data from yfinance (limitations or network). "
                "Falling back to 7 days...",
                lookback_to_try,
            )
            lookback_to_try = 7
            all_ticks = self._download_ticks_for_period(lookback_to_try)

        if not all_ticks:
            raise RuntimeError("No ticks fetched — check tickers and internet access")

        all_ticks.sort(key=lambda t: t.timestamp)
        return all_ticks

    def _download_ticks_for_period(self, days: int) -> List[MarketTick]:
        """Download ticks for a specific period in days."""
        all_ticks: List[MarketTick] = []
        period = f"{days}d"

        for symbol in self.tickers:
            try:
                df: pd.DataFrame = yf.download(
                    tickers=symbol,
                    period=period,
                    interval="1m",
                    progress=False,
                    auto_adjust=True,
                )
                if df.empty:
                    logger.warning("⚠️  No data returned for %s with period %s", symbol, period)
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                for ts, row in df.iterrows():
                    try:
                        # Ensure timezone-aware UTC timestamp
                        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                            utc_ts = ts.to_pydatetime().astimezone(timezone.utc)
                        else:
                            utc_ts = ts.to_pydatetime().replace(tzinfo=timezone.utc)

                        tick = MarketTick(
                            symbol=symbol,
                            timestamp=utc_ts,
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                            volume=float(row["volume"]),
                        )
                        all_ticks.append(tick)
                    except Exception as exc:
                        logger.debug("Skipping malformed row %s/%s: %s", symbol, ts, exc)

                logger.info("  ✔ %s: %d ticks loaded with period %s", symbol, len(df), period)

            except Exception as exc:
                logger.error("Failed to fetch %s with period %s: %s", symbol, period, exc)

        return all_ticks


    def _publish(self, tick: MarketTick) -> None:
        """
        Serialise and publish a single MarketTick to Kafka.

        Uses the ticker symbol as the message key for partition affinity
        (all ticks for one symbol go to the same partition, preserving order).
        """
        try:
            self._producer.produce(
                topic=self.topic,
                key=tick.symbol.encode("utf-8"),
                value=tick.to_json().encode("utf-8"),
                on_delivery=self._delivery_callback,
            )
            # Poll to trigger delivery callbacks without blocking
            self._producer.poll(0)
        except BufferError:
            logger.warning("Kafka producer queue full — flushing before retry")
            self._producer.flush()
            self._producer.produce(
                topic=self.topic,
                key=tick.symbol.encode("utf-8"),
                value=tick.to_json().encode("utf-8"),
                on_delivery=self._delivery_callback,
            )

    @staticmethod
    def _delivery_callback(err: Optional[KafkaError], msg: object) -> None:
        """confluent-kafka delivery report callback."""
        if err:
            logger.error("❌ Delivery failed: %s", err)
        else:
            logger.debug(
                "✅ Delivered → topic=%s partition=%s offset=%s",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )

    cfg = get_settings()
    producer = KafkaMarketProducer(
        tickers=cfg.tickers_list,
        topic=cfg.kafka_topic_market_data,
        bootstrap_servers=cfg.kafka_bootstrap_servers,
        interval_seconds=cfg.stream_interval_seconds,
        lookback_days=cfg.lookback_days,
        loop=True,
    )
    try:
        producer.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user — shutting down")
    finally:
        producer.close()
