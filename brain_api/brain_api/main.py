"""FastAPI application entrypoint."""

from fastapi import FastAPI

from brain_api.routes import health, root

app = FastAPI(
    title="Brain API",
    description="FastAPI brain service for LearnFinance-2025",
    version="0.1.0",
)

app.include_router(root.router)
app.include_router(health.router, prefix="/health", tags=["health"])

