ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install dependencies first (better layer caching)
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# Copy application code
COPY . .

EXPOSE 8080

# Serve the FastAPI app; APScheduler starts in the lifespan hook.
ENTRYPOINT ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
