# Dockerfile — Credit Card Fraud Detection API
# Base: python:3.12-slim (Debian, no dev tools = smaller image)
# Target: Hugging Face Spaces (Docker SDK)
#
# Sources:
#   HF Spaces Docker SDK: https://huggingface.co/docs/hub/spaces-sdks-docker
#   HF Spaces config ref: https://huggingface.co/docs/hub/spaces-config-reference
#   Docker best practices: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
#   gunicorn deployment:   https://docs.gunicorn.org/en/stable/deploy.html

# ── Stage 1: Base ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE=1  → no .pyc files (saves space)
# PYTHONUNBUFFERED=1         → stdout/stderr not buffered (HF Logs need this)
# PORT                       → HF Spaces routes traffic to app_port (7860) from README YAML;
#                              the app reads PORT, so we set it to match.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Create the same UID (1000) that HF Spaces uses when running containers
# Source: https://huggingface.co/docs/hub/spaces-sdks-docker#permissions
RUN useradd -m -u 1000 user

# Set working directory (HF convention: /app under the user's home)
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────────────────
# Copy requirements first (Docker layer cache: if requirements unchanged,
# this layer is reused and pip install is skipped on rebuild)
COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY --chown=user app.py .
COPY --chown=user model/ ./model/

# Switch to non-root user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Expose port ───────────────────────────────────────────────────────────────
# HF Spaces routes external traffic to the port set via app_port in README.md
# (default 7860). EXPOSE is documentation only.
EXPOSE 7860

# ── Health check ─────────────────────────────────────────────────────────────
# Optional locally; HF Spaces uses its own readiness probe against app_port.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

# ── Startup command ───────────────────────────────────────────────────────────
# gunicorn is the production WSGI server — never use flask's built-in server in prod
# --workers 2     → 2 worker processes (HF free CPU tier: 2 vCPU; 2 workers is safe)
# --threads 2     → 2 threads per worker (handles concurrent requests)
# --timeout 120   → 120s timeout (XGBoost inference is fast, plenty of headroom)
# --bind          → binds to $PORT (7860, matching app_port in README.md)
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
