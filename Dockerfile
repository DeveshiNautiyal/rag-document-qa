FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS build

WORKDIR /app

# Copy manifest files and install dependencies into /app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --locked --no-dev

# ── Runtime stage ─────────────────────────────────────────────────────────────
# Same base as the build stage: the venv's python binary is a symlink into
# this image's system Python install, so the tag must match across stages.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS runtime

WORKDIR /app

# Install only the shared libraries that the Python ecosystem needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the populated virtual environment and source from the build stage
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
COPY --from=build /app/pyproject.toml /app/pyproject.toml

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "simple_rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
