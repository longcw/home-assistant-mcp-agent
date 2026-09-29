ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# git, for the dependencies pyproject.toml pins to commits; patch, for patches/
RUN apt-get update && apt-get install -y --no-install-recommends git patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Install dependencies first (better layer caching)
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen

# fixes not yet in the pinned livekit-agents, applied to the installed package; a patch
# that stops applying fails the build. agents-7542-mcp.patch: livekit/agents#7542, which
# tells an MCP server when a tool call is cancelled
COPY patches/ patches/
RUN site=$(uv run --no-sync python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && for p in patches/*.patch; do patch -p1 -d "$site" < "$p"; done

# Copy application code
COPY . .

# Ensure any dependent model files are downloaded at build time (no-op for the
# inference-based pipeline, but future-proofs local plugins)
RUN uv run -m livekit.agents download-files

# Run the agent worker (sources live under src/; prompt.yaml stays at /app)
ENTRYPOINT ["uv", "run", "src/main.py"]
CMD ["start"]
