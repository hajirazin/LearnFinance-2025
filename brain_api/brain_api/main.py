"""FastAPI application entrypoint."""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Configure logging to show INFO level logs from our modules
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# Load .env file before other imports that may read environment variables
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from brain_api.routes import (
    allocation,
    alpaca,
    email,
    etl,
    experience,
    health,
    inference,
    llm,
    models_meta,
    orders,
    root,
    signals,
    training,
    universe,
)

shutdown_event = threading.Event()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Manage app lifecycle: signal background tasks on shutdown."""
    yield
    logger.info("Shutdown requested, signalling background tasks to stop...")
    shutdown_event.set()


app = FastAPI(
    title="Brain API",
    description="FastAPI brain service for LearnFinance-2025",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Log validation errors with details for debugging."""
    logger.error(f"[Validation] Request to {request.url.path} failed validation:")
    for error in exc.errors():
        logger.error(f"  - {error['loc']}: {error['msg']} (type={error['type']})")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


app.include_router(root.router)
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(alpaca.router, prefix="/alpaca", tags=["alpaca"])
app.include_router(universe.router, prefix="/universe", tags=["universe"])
app.include_router(training.router, prefix="/train", tags=["training"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])
app.include_router(signals.router, prefix="/signals", tags=["signals"])
app.include_router(allocation.router, prefix="/allocation", tags=["allocation"])
app.include_router(etl.router, prefix="/etl", tags=["etl"])
app.include_router(experience.router, prefix="/experience", tags=["experience"])
app.include_router(orders.router, prefix="/orders", tags=["orders"])
app.include_router(llm.router, prefix="/llm", tags=["llm"])
app.include_router(email.router, prefix="/email", tags=["email"])
app.include_router(models_meta.router, prefix="/models", tags=["models"])
