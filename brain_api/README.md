# Brain API

FastAPI service for LearnFinance-2025. This is the Python "brain" that will handle ML inference, training, and evidence synthesis.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
cd brain_api
uv sync --all-extras
```

## Run locally

```bash
uv run uvicorn brain_api.main:app --reload --port 8000
```

The API will be available at http://localhost:8000

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Hello world |
| GET | `/health` | Generic health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |

## API documentation

Once running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Run tests

```bash
uv run pytest
```


