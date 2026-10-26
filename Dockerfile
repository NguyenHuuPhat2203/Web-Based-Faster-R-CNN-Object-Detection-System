# syntax=docker/dockerfile:1.7

# ── Build frontend ────────────────────────────────────────
FROM node:20-alpine@sha256:fb4cd12c85ee03686f6af5362a0b0d56d50c58a04632e6c0fb8363f609372293 AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY frontend/ ./
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# ── Build backend ─────────────────────────────────────────
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea AS backend-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt ./backend-requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r backend-requirements.txt

COPY backend/ /app/backend/
COPY ml/ /app/ml/
ENV PYTHONPATH=/app

# ── Runtime ───────────────────────────────────────────────
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 nginx supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /etc/nginx/conf.d/default.conf

WORKDIR /app

# Backend deps
COPY --from=backend-build /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=backend-build /usr/local/bin/ /usr/local/bin/

# Model weights at the Hub commit pinned in models.lock.json; the Space secret
# HF_TOKEN reads the private model repo. Placed before the app code so
# code-only pushes reuse this layer.
COPY scripts/sync_models.py /app/scripts/
COPY models.lock.json /app/
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    HF_TOKEN="$(cat /run/secrets/HF_TOKEN)" HF_HOME=/tmp/hf \
    python scripts/sync_models.py pull \
    && rm -rf /tmp/hf /app/.cache

COPY backend/ /app/backend/
COPY ml/ /app/ml/
COPY registry/ /app/registry/

RUN mkdir -p /app/uploads

# Copy frontend
COPY --from=frontend-build /app/frontend/dist /usr/share/nginx/html

# Nginx config — serve frontend + proxy API to backend + expose healthz
RUN cat > /etc/nginx/conf.d/hf.conf <<'NGINX'
server {
    listen 7860;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    location = /healthz {
        proxy_pass http://127.0.0.1:8000/healthz;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30s;
    }

    # API proxy
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        client_max_body_size 50M;
    }


    location / {
        try_files $uri $uri/ /index.html;
        add_header Cache-Control "no-cache, must-revalidate";
    }
}
NGINX

# Supervisor config — run nginx + uvicorn
RUN cat > /etc/supervisor/conf.d/app.conf <<'SUPERVISOR'
[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:uvicorn]
command=uvicorn backend.main:app --host 0.0.0.0 --port 8000
directory=/app
environment=PYTHONPATH="/app"
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
SUPERVISOR

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 7860

RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    '' \
    'alembic -c /app/backend/alembic.ini upgrade head' \
    'exec supervisord -c /etc/supervisor/supervisord.conf -n' \
    > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

CMD ["/usr/local/bin/start.sh"]
