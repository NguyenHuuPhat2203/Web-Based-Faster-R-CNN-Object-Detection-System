import os

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config as cfg
from database import engine, Base
from routers import auth, predict, predict_yolo, images

# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(title="Brain Tumor Detector")

# CORS – allow the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001", "http://127.0.0.1:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ──────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)

# ── Routers ───────────────────────────────────────────────────────────

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(predict_yolo.router)
app.include_router(images.router)

# Optional – serve uploaded images statically so the frontend can display them
if os.path.exists(cfg.UPLOAD_DIR):
    app.mount("/static/uploads", StaticFiles(directory=cfg.UPLOAD_DIR), name="uploads")


# ── Health ────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "status": "Backend is running",
        "device": device,
        "auth": "enabled",
    }
