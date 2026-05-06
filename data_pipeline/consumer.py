"""
data_pipeline/consumer.py — KafkaMarketConsumer

Reads raw market tick messages from the Kafka topic, deserialises them,
runs DataValidator, and dispatches ValidatedTick objects to a user-supplied
callback function for downstream processing (feature engineering, etc.).

Design decisions:
  - Blocking poll loop — intended to run in its own process or thread.
  - DataValidator is stateful (per-symbol rolling close); one instance shared
    across all messages in the same consumer process.
  - Invalid ticks are logged and counted but NOT forwarded to the callback,
    keeping the feature engineering layer clean.
  - Graceful shutdown via SIGINT / KeyboardInterrupt.
"""
from __future__ import annotations

import logging
import sys
from typing import Callable, Optional

from confluent_kafka import Consumer as ConfluentConsumer
from confluent_kafka import KafkaError, KafkaException, Message

from config.settings import get_settings
from data_pipeline.schemas import MarketTick, ValidatedTick
from data_pipeline.validator import DataValidator

logger = logging.getLogger(__name__)


class KafkaMarketConsumer:
    """
    Consumes OHLCV ticks from Kafka, validates them, and forwards valid
    ticks to a downstream callback.

    Args:
        topic:              Kafka topic to subscribe to.
        bootstrap_servers:  Kafka broker address(es).
        group_id:           Consumer group ID.
        security_protocol:  Kafka security protocol (default PLAINTEXT).
        on_valid_tick:      Callback invoked with each ValidatedTick that
                            passes all DataValidator checks.
        on_invalid_tick:    Optional callback for quarantined ticks.
        poll_timeout:       Kafka poll timeout in seconds (default 1.0).
        validator:          DataValidator instance; a default one is created
                            if not supplied.
    """

    def __init__(
        self,
        topic: str,
        bootstrap_servers: str,
        group_id: str,
        security_protocol: str = "PLAINTEXT",
        on_valid_tick: Optional[Callable[[ValidatedTick], None]] = None,
        on_invalid_tick: Optional[Callable[[ValidatedTick], None]] = None,
        poll_timeout: float = 1.0,
        validator: Optional[DataValidator] = None,
    ) -> None:
        self.topic = topic
        self.poll_timeout = poll_timeout
        self.on_valid_tick = on_valid_tick or self._default_valid_handler
        self.on_invalid_tick = on_invalid_tick or self._default_invalid_handler
        self.validator = validator or DataValidator()

        self._consumer = ConfluentConsumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "security.protocol": security_protocol,
                "group.id": group_id,
                "auto.offset.reset": "latest",
                "enable.auto.commit": True,
                "session.timeout.ms": 30_000,
                "heartbeat.interval.ms": 10_000,
            }
        )
        self._running = False
        self._stats = {"received": 0, "valid": 0, "invalid": 0, "errors": 0}

        logger.info(
            "KafkaMarketConsumer initialised — brokers=%s topic=%s group=%s",
            bootstrap_servers,
            topic,
            group_id,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Start the blocking poll loop.

        Subscribes to the Kafka topic and continuously polls for messages.
        Blocks until stop() is called or KeyboardInterrupt is raised.
        """
        self._consumer.subscribe([self.topic])
        self._running = True
        logger.info("▶  Consumer started — subscribed to '%s'", self.topic)

        try:
            while self._running:
                msg: Optional[Message] = self._consumer.poll(self.poll_timeout)

                if msg is None:
                    continue  # Timeout — no messages in this window

                if msg.error():
                    self._handle_kafka_error(msg)
                    continue

                self._process_message(msg)

        except KeyboardInterrupt:
            logger.info("⛔ Consumer interrupted by user")
        finally:
            self._consumer.close()
            self._log_stats()

    def stop(self) -> None:
        """Signal the poll loop to exit cleanly on next iteration."""
        self._running = False
        logger.info("KafkaMarketConsumer stop requested")

    @property
    def stats(self) -> dict:
        """Return a copy of the current message counters."""
        return dict(self._stats)

    # ── Private helpers ───────────────────────────────────────────────────

    def _process_message(self, msg: Message) -> None:
        """Deserialise, validate, and dispatch a single Kafka message."""
        self._stats["received"] += 1
        try:
            tick = MarketTick.from_json(msg.value())
            validated = self.validator.validate(tick)

            if validated.is_valid:
                self._stats["valid"] += 1
                self.on_valid_tick(validated)
            else:
                self._stats["invalid"] += 1
                self.on_invalid_tick(validated)

        except Exception as exc:
            self._stats["errors"] += 1
            logger.error(
                "Failed to process message offset=%s: %s",
                msg.offset(),
                exc,
                exc_info=True,
            )

    def _handle_kafka_error(self, msg: Message) -> None:
        """Handle Kafka-level errors returned in message.error()."""
        err = msg.error()
        if err.code() == KafkaError._PARTITION_EOF:
            logger.debug(
                "End of partition reached %s [%d] offset %d",
                msg.topic(),
                msg.partition(),
                msg.offset(),
            )
        else:
            self._stats["errors"] += 1
            logger.error("Kafka error: %s", err)

    def _log_stats(self) -> None:
        """Log final consumption statistics."""
        s = self._stats
        logger.info(
            "Consumer stats — received=%d valid=%d invalid=%d errors=%d",
            s["received"],
            s["valid"],
            s["invalid"],
            s["errors"],
        )

    @staticmethod
    def _default_valid_handler(tick: ValidatedTick) -> None:
        """Default callback: log valid ticks to stdout."""
        print(
            f"[VALID]  {tick.symbol:<6} "
            f"ts={tick.timestamp.strftime('%Y-%m-%d %H:%M')}  "
            f"close={tick.close:>10.4f}  vol={tick.volume:>12,.0f}"
        )

    @staticmethod
    def _default_invalid_handler(tick: ValidatedTick) -> None:
        """Default callback: log invalid ticks as warnings."""
        logger.warning(
            "[INVALID] %s — %s", tick.symbol, " | ".join(tick.validation_errors)
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
    )

    cfg = get_settings()
    consumer = KafkaMarketConsumer(
        topic=cfg.kafka_topic_market_data,
        bootstrap_servers=cfg.kafka_bootstrap_servers,
        group_id=cfg.kafka_group_id,
        security_protocol=cfg.kafka_security_protocol,
    )
    consumer.run()
