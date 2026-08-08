# Architecture

Pipeline Clean Architecture for brain-tumor detection (ml_rules_en).

## Layout

```
ml/                 # shared kernel (no FastAPI, no train loop)
  backbone/         # Faster R-CNN factory + YOLO thin wrapper
  core/             # entities, constants, pure postprocess
  config/           # shared Pydantic Model/Data/Train configs
  transforms/       # inference transforms (no medical color jitter)
backend/            # inference shell (Python package; was backend/app)
  pipeline/         # FRCNN/YOLO run + GradCAM + OOM handling
  api/              # v1 bare + v2 item-schema
  services/         # auth, persistence, prediction orchestration
training/           # training shell (scripts, trainer, data)
registry/           # promoted artifacts + metadata.json
```

## Import law

- `backend` → `ml.*`
- `training` → `ml.*`
- **backend must not import training**
- **training must not import backend**

Run Python from the **repo root** so `ml` and `backend` resolve via cwd (no local wheel / no `sys.path` in `.py`). Docker sets `PYTHONPATH=/app` with the same tree (`/app/ml`, `/app/backend`).

## API

- **v1** `/predict`, `/predict-yolo` — bare `PredictionResult`
- **v2** `/v2/predict`, `/v2/predict/yolo` — `{status, result|error}`; GradCAM via `?explain=true`
- Frontend defaults to v2 (`VITE_API_VERSION`); rollback with `v1`

## Artifacts

- Prefer `registry/faster_rcnn/<ver>/model.safetensors`
- YOLO keeps `.pt` under `registry/yolo/<ver>/`
- Legacy `checkpoints/*.pth` + HuggingFace fallback during transition
- Promote helper: `python -m training.scripts.promote_from_hf`
- Weights are never baked into `Dockerfile.backend`; mount `./registry` and `./checkpoints` via compose

## Serve (Docker Compose)

```
./registry      →  /registry      (REGISTRY_DIR)
./checkpoints   →  /checkpoints   (CHECKPOINT_DIR)
frontend waits until backend /healthz is healthy
```
