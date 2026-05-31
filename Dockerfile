# ── Build frontend ────────────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci && npm cache clean --force

COPY frontend/ .
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# ── Build backend ─────────────────────────────────────────
FROM python:3.12-slim AS backend-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Runtime ───────────────────────────────────────────────
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 nginx supervisor \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /etc/nginx/conf.d/default.conf

WORKDIR /app

# Copy backend
COPY --from=backend-build /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=backend-build /usr/local/bin/ /usr/local/bin/
COPY backend/ /app/
RUN mkdir -p /checkpoints /app/uploads

# Copy frontend
COPY --from=frontend-build /app/dist /usr/share/nginx/html

# Nginx config — serve frontend + proxy API to backend
RUN cat > /etc/nginx/conf.d/hf.conf <<'NGINX'
server {
    listen 7860;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

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
command=uvicorn main:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
stderr_logfile=/dev/stderr
stderr_logfile_maxbytes=0
SUPERVISOR

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf", "-n"]
