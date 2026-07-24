FROM python:3.11-slim
WORKDIR /app
RUN pip install uv

# 1. Copy ONLY the files that list your dependencies
COPY pyproject.toml ./

COPY uv.lock ./

# 2. Sync the dependencies ONLY (ignore the project code for now)
RUN uv sync --frozen --no-dev --no-install-project

# 3. Now copy the rest of your application code
COPY . .

# 4. Sync again to install your actual project code
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "uvicorn", "brain_api.main:app", "--host", "0.0.0.0", "--port", "8000"]