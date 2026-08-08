# Model Registry

Promoted artifacts live here. Do not deploy directly from `checkpoints/` or `training/outputs/`.

Weights (`.safetensors`, `.pt`, `.pth`) are **gitignored**. Commit `metadata.json` only.

## Layout

```
registry/
├── faster_rcnn/<version>/
│   ├── model.safetensors   # gitignored
│   └── metadata.json
└── yolo/<version>/
    ├── yolov8_tumor.pt     # gitignored
    └── metadata.json
```

## One-shot promote from Hugging Face

Downloads FRCNN `model.pth` into `checkpoints/`, converts to safetensors, writes registry metadata.
Promotes YOLO only if `checkpoints/yolov8_tumor.pt` already exists.

```bash
python training/scripts/promote_from_hf.py
# or reuse a local pth:
python training/scripts/promote_from_hf.py --skip-download
```

## Manual promote

```bash
python training/scripts/export_safetensors.py \
  --checkpoint checkpoints/model.pth \
  --version v1.0.0 \
  --dataset-version v1 \
  --dataset-hash <sha256> \
  --map50 0.69 --map5095 0.49

python training/scripts/promote_yolo.py \
  --checkpoint checkpoints/yolov8_tumor.pt \
  --version v1.0.0 \
  --dataset-version v1 \
  --dataset-hash <sha256>
```

## Docker / Compose

The backend image is **thin** (no baked weights). Mount host artifacts:

```bash
docker compose up --build
```

Volumes (see `docker-compose.yml`):

- `./registry` → `/registry` (`REGISTRY_DIR`)
- `./checkpoints` → `/checkpoints` (`CHECKPOINT_DIR`)

At runtime the API prefers registry safetensors, then local checkpoints, then Hugging Face download fallback.
