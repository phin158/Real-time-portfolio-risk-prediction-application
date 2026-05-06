"""
api/main.py — FastAPI Application Entrypoint.

Handles lifespan events to start the Kafka Consumer background thread
and initialize the RiskService (which loads the PyTorch model).
"""
from contextlib import asynccontextmanager
import threading
import logging
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from api.routers import portfolio, websocket
from api.services.risk_service import RiskService
from data_pipeline.consumer import KafkaMarketConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
risk_service: RiskService | None = None
consumer: KafkaMarketConsumer | None = None
consumer_thread: threading.Thread | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global risk_service, consumer, consumer_thread
    cfg = get_settings()
    
    logger.info("Starting up FastAPI application...")
    
    # 1. Initialize RiskService (loads PyTorch model, sets up FeatureEngineer and Redis)
    risk_service = RiskService()
    
    # 2. Start Kafka Consumer in a background thread
    logger.info("Starting KafkaConsumer background thread...")
    consumer = KafkaMarketConsumer(
        topic=cfg.kafka_topic_market_data,
        bootstrap_servers=cfg.kafka_bootstrap_servers,
        group_id="fastapi_backend_group",
        on_valid_tick=risk_service.process_tick
    )
    
    consumer_thread = threading.Thread(target=consumer.run, daemon=True)
    consumer_thread.start()
    
    yield
    
    # 3. Shutdown gracefully
    logger.info("Shutting down FastAPI application...")
    if consumer:
        consumer.stop()
    if consumer_thread:
        consumer_thread.join(timeout=5.0)

app = FastAPI(
    title="Real-time Portfolio Risk API",
    description="Backend for streaming VaR and Volatility predictions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio.router, prefix="/api", tags=["portfolio"])
app.include_router(websocket.router, tags=["websocket"])

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
