# syntax=docker/dockerfile:1.7

# ── Build frontend ────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY frontend/ ./
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

# ── Build backend ─────────────────────────────────────────
FROM python:3.12-slim AS backend-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY backend/ /app/

# Download model checkpoint from Hugging Face Hub at build time
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='LuxeFats/FasterRCNN-Checkpoint', filename='model.pth', local_dir='/checkpoints')" 2>/dev/null || echo "Checkpoint download skipped (will download at runtime)"

# ── Runtime ───────────────────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 nginx supervisor curl cron \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /etc/nginx/conf.d/default.conf

WORKDIR /app

# Copy backend
COPY --from=backend-build /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=backend-build /usr/local/bin/ /usr/local/bin/
COPY backend/ /app/

# Copy pre-downloaded checkpoint from builder (if any)
COPY --from=backend-build /checkpoints/ /checkpoints/
RUN mkdir -p /checkpoints /app/uploads /var/log/cron

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

    location /static/uploads/ {
        alias /app/uploads/;
    }

    location / {
        try_files $uri $uri/ /index.html;
        add_header Cache-Control "no-cache, must-revalidate";
    }
}
NGINX

# Supervisor config — run nginx + uvicorn + cron
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
command=uvicorn main:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0

[program:cron]
command=/usr/sbin/cron -f
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
SUPERVISOR

# Hourly health check ping against the public endpoint
RUN cat > /etc/cron.d/healthz <<'CRON'
SHELL=/bin/sh
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
0 * * * * root curl -fsS http://127.0.0.1:7860/healthz >/var/log/healthz.log 2>&1 || true
CRON
RUN chmod 0644 /etc/cron.d/healthz

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    '' \
    'alembic -c /app/alembic.ini upgrade head' \
    'exec supervisord -c /etc/supervisor/supervisord.conf -n' \
    > /usr/local/bin/start.sh && chmod +x /usr/local/bin/start.sh

CMD ["/usr/local/bin/start.sh"]
