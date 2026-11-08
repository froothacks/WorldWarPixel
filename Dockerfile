# Container image for the Cloud Run deployment (deploy/cloudrun.sh). Mirrors deploy/modal_app.py:
# the frontend is built in and deploy/serve.py runs uvicorn with the SQLite restore/backup loop.
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build

FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_NO_DEV=1
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY src ./src
COPY deploy/serve.py ./deploy/serve.py
COPY --from=frontend /app/frontend/dist ./frontend/dist
# The package runs from source (like the Modal image) so app.py finds ../frontend/dist.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/src \
    EVENT_LOG_PATH=/root/state/events.db \
    EVENT_LOG_BACKUP=/data/events.db \
    BACKUP_INTERVAL=60 \
    SHUTDOWN_GRACE=4 \
    TIMELAPSE_DIR=/data/timelapses \
    PORT=8080
EXPOSE 8080
CMD ["python", "deploy/serve.py"]
