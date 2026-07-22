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
RUN uv sync --no-dev --frozen

# Copy application code
COPY . .

# Ensure any dependent model files are downloaded at build time (no-op for the
# inference-based pipeline, but future-proofs local plugins)
RUN uv run -m livekit.agents download-files

# Run the agent worker (sources live under src/; prompt.yaml stays at /app)
ENTRYPOINT ["uv", "run", "src/main.py"]
CMD ["start"]
