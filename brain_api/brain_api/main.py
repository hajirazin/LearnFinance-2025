"""FastAPI application entrypoint."""

import logging

from dotenv import load_dotenv

# Configure logging to show INFO level logs from our modules
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

# Load .env file before other imports that may read environment variables
load_dotenv()

from fastapi import FastAPI

from brain_api.routes import (
    allocation,
    etl,
    health,
    inference,
    root,
    signals,
    training,
    universe,
)

app = FastAPI(
    title="Brain API",
    description="FastAPI brain service for LearnFinance-2025",
    version="0.1.0",
)

app.include_router(root.router)
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(universe.router, prefix="/universe", tags=["universe"])
app.include_router(training.router, prefix="/train", tags=["training"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])
app.include_router(signals.router, prefix="/signals", tags=["signals"])
app.include_router(allocation.router, prefix="/allocation", tags=["allocation"])
app.include_router(etl.router, prefix="/etl", tags=["etl"])
