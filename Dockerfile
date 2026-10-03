# ==============================================================================
# Multi-stage Production Dockerfile for Sentiment Analysis Service
# ==============================================================================

# -----------------------------
# Stage 1: Build Dependencies
# -----------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY fastapi_app/requirements.txt /build/requirements.txt
RUN pip install --no-cache-dir --user -r /build/requirements.txt

# Pre-download NLTK data to eliminate runtime download delays
RUN python -m nltk.downloader -d /build/nltk_data stopwords wordnet

# -----------------------------
# Stage 2: Production Runtime
# -----------------------------
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    NLTK_DATA=/app/nltk_data \
    PATH=/home/appuser/.local/bin:$PATH

# Install curl for container health check probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root user for security compliance
RUN useradd -u 1000 -m -s /bin/bash appuser

# Copy installed Python packages and NLTK data from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /build/nltk_data /app/nltk_data

# Copy application source and model artifacts
COPY fastapi_app/ /app/fastapi_app/
COPY models/ /app/models/

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Docker healthcheck for container orchestrators (Azure App Service / ACA / K8s)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Production startup command using Uvicorn ASGI server
CMD ["sh", "-c", "uvicorn fastapi_app.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
