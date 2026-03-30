# ── Stage 1: Development ─────────────────────────────────────────────────────
FROM python:3.12-slim AS development

WORKDIR /app

# System dependencies for psycopg, PDF parsing (unstructured), and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libc6-dev zlib1g-dev libpq-dev curl \
        libxcb1 libx11-6 libgl1 libglib2.0-0 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[dev]"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ── Stage 2: Production ──────────────────────────────────────────────────────
FROM python:3.12-slim AS production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir .

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
