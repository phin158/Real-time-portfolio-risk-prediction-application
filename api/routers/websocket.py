"""
api/routers/websocket.py — WebSocket endpoint subscribing to Redis Pub/Sub.

Clients connect to /ws/risk-stream to receive realtime predictions.
"""
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from config.settings import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/risk-stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cfg = get_settings()

    # Async Redis client for the ASGI loop
    redis_client = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
    pubsub = redis_client.pubsub()

    try:
        # Use cfg.redis_channel — not hardcoded — for consistency with RiskService
        await pubsub.subscribe(cfg.redis_channel)
        logger.info(
            "WebSocket client connected and subscribed to Redis channel '%s'",
            cfg.redis_channel,
        )

        while True:
            # Check for disconnects by pinging client
            # But get_message blocks, so we use timeout and sleep
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message is not None:
                await websocket.send_text(message["data"])
            else:
                # Small sleep to yield to event loop
                await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        await pubsub.unsubscribe(cfg.redis_channel)
        await redis_client.aclose()
