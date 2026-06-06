# Dockerfile — Credit Card Fraud Detection API
# Base: python:3.12-slim (Debian Bullseye, no dev tools = smaller image)
# Target: Azure Web App for Containers
#
# Sources:
#   Azure Web App for Containers:
#     https://learn.microsoft.com/en-us/azure/app-service/configure-custom-container
#   Docker best practices:
#     https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
#   gunicorn deployment:
#     https://docs.gunicorn.org/en/stable/deploy.html

# ── Stage 1: Base ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE=1  → no .pyc files (saves space)
# PYTHONUNBUFFERED=1         → stdout/stderr not buffered (Azure Log Stream needs this)
# PORT                       → Azure sets this automatically; default 8000
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Create a non-root user for security
# Azure Web App runs as root by default, but non-root is best practice
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy requirements first (Docker layer cache: if requirements unchanged,
# this layer is reused and pip install is skipped on rebuild)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY app.py .
COPY model/ ./model/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# ── Expose port ───────────────────────────────────────────────────────────────
# EXPOSE is documentation only — Azure Web App ignores it and uses PORT env var
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Docker and Azure use this to know when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# ── Startup command ───────────────────────────────────────────────────────────
# gunicorn is the production WSGI server — never use flask's built-in server in prod
# --workers 2     → 2 worker processes (Azure B1 tier: 1 vCPU, 1.75 GB RAM; 2 is safe)
# --threads 2     → 2 threads per worker (handles concurrent requests)
# --timeout 120   → 120s timeout (XGBoost inference is fast, but shap can be slow)
# --bind          → Azure sets PORT env var; we bind to it
# app:app         → module 'app', Flask instance 'app'
#
# Source: https://docs.gunicorn.org/en/stable/settings.html
CMD gunicorn \
    --bind 0.0.0.0:${PORT} \
    --workers 2 \
    --threads 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    app:app
