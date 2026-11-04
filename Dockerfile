# NCAA Bayesian Basketball: one image for the web app and the background worker.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    NCAA_SEASON=2026 \
    NCAA_LIVE_SEASON=2027

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt gunicorn==23.0.0

COPY ncaa ncaa
COPY webapp webapp
COPY data data
COPY deploy/worker.sh deploy/worker.sh
RUN chmod +x deploy/worker.sh && mkdir -p artifacts

EXPOSE 8000
# Web by default; the worker service overrides this with deploy/worker.sh.
# One process with threads: live-game history is kept in memory, so it must be shared.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "600", "--bind", "0.0.0.0:8000", "webapp.app:app"]
