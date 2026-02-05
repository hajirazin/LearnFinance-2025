FROM python:3.11-slim
WORKDIR /app
RUN pip install uv
COPY . .
RUN uv sync --frozen --no-dev
CMD ["uv", "run", "uvicorn", "brain_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
