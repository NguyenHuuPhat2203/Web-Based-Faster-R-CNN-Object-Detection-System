---
title: Brain Tumor Detector
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# Brain Tumor Detector

End-to-end brain-tumor analysis on MRI:

- 2D detection on uploaded slices with Faster R-CNN (ResNet-50 FPN) or YOLOv8,
  optional Grad-CAM explanation, scan history, and PDF reports.
- 3D case viewer for BraTS 2021 cases with expert masks and on-demand
  SwinUNETR segmentation (NCR / ED / ET volumes).

Stack: PyTorch, FastAPI, SQLAlchemy/Alembic, React + Vite + TypeScript.

## Results

Faster R-CNN on the validation split after 10 training epochs:

- mAP@0.5: **0.69**
  ![mAP@0.5](figures/mAP50.png)
- mAP@0.5:0.95: **0.49**
  ![mAP@0.5:0.95](figures/mAP5090.png)

## Dataset

- Ultralytics brain-tumor dataset
  (https://docs.ultralytics.com/datasets/detect/brain-tumor/), YOLO format.
- 1116 images: 893 train, 223 validation.
- Classes: `0 = negative`, `1 = positive`. Faster R-CNN reserves label 0 for
  background, so its labels are `1 = negative`, `2 = positive`. Both models
  report the names `negative` / `positive`.
- Download and normalize into `data/raw/v1/` (gitignored):
  `python -m training.scripts.download_dataset`.
- BraTS 2021 cases for the 3D viewer live in `data/BraTS2021_Training_Data/`.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for layers, import rules, API, and
naming conventions.

```
frontend (React) --HTTP--> backend (FastAPI) --> ml (models, core) --> weights
                                   |                                 (registry/,
                                   +--> SQLite + uploads/             checkpoints/,
training (scripts) --> ml ---------------------------> registry/      .weights/)
```

## Project structure

```
ml/                  shared model kernel (no FastAPI, no training loop)
  core/              entities, constants, labels, post-processing
  models/            faster_rcnn.py, yolo.py, swin_unetr.py
  transforms/        inference image transforms
  config/            shared Pydantic training configs
backend/             FastAPI application package
  api/routers/       auth, predict (v1 + v2), images, case_viewer
  pipeline/          inference engine, model adapters, Grad-CAM
  services/          business logic (prediction, persistence, reports, cases)
  repositories/      database access
  models/            SQLAlchemy ORM models
  schemas/           Pydantic request/response models
  alembic/           database migrations
training/            training, evaluation, export, promotion
  data/  trainer/  scripts/  notebooks/  tests/
registry/            promoted model weights + metadata.json
scripts/             repo-level tools: sync_models.py (weights <-> Hugging Face)
frontend/            React application
```

## Setup

Run every Python command from the repository root so `ml`, `backend`, and
`training` import from the working directory. Do not `pip install -e .`.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
```

Create `backend/.env` from `backend/.env.example` and set `SECRET_KEY` to a
random value of at least 32 characters; the API refuses to start otherwise:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Model weights

| Model | Path | Produced by |
|---|---|---|
| Faster R-CNN | `registry/faster_rcnn/<version>/model.safetensors` (falls back to `checkpoints/model.safetensors`, then `checkpoints/model.pth`) | `training.scripts.promote_checkpoint` or `export_safetensors` |
| YOLOv8 | `registry/yolo/<version>/yolov8_tumor.pt` (falls back to `checkpoints/yolov8_tumor.pt`) | `training.scripts.promote_yolo` |
| SwinUNETR | `.weights/fold0_f48_ep300_4gpu_dice0_8854/model.pt` (`SWIN_UNETR_WEIGHTS`) | MONAI research-contributions BRATS21 fold-0 release |
| MedNeXt-XS (experiment) | `.weights/mednext_xs/mednext_xs_best.pth` | `training/notebooks/mednext_xs_brats.ipynb` |

All of these files live in one Hugging Face model repo at the same paths;
`models.lock.json` pins the commit that matches the code. Copy `.env.example`
to `.env` and set `HF_TOKEN` and `HF_MODEL_REPO_ID`, then:

```bash
python -m scripts.sync_models pull    # download the pinned weights
python -m scripts.sync_models push    # upload changed weights, update the pin
```

Missing weights are reported as errors; the API never downloads weights at
runtime.

## Run

Backend (http://localhost:8000):

```bash
alembic -c backend/alembic.ini upgrade head
python -m uvicorn backend.main:app --reload
```

Frontend (http://localhost:8001):

```bash
cd frontend
npm install
npm run dev
```

Docker Compose (thin backend image; weights mounted from the host, so pull
them first):

```bash
python -m scripts.sync_models pull
docker compose up --build
```


## Training

```bash
python -m training.scripts.run_e2e_smoke               # download, subsample, 1-epoch smoke
python -m training.scripts.train_faster_rcnn           # full Faster R-CNN run on data/raw/v1
python -m training.scripts.train_yolo                  # full YOLOv8 run
python -m training.scripts.promote_checkpoint --model faster_rcnn \
  --checkpoint training/outputs/faster_rcnn/best_model.pth --version v1.1.0
```

Promotion refuses to overwrite `v1.0.0` without `--i-know`. Archived
experiment notebooks live in `training/notebooks/`.

## Checks

```bash
ruff check backend ml training scripts && ruff format --check backend ml training scripts
pytest -m "not overfit and not perf"
cd frontend && npx tsc --noEmit && npm test && npm run build
```

## Author

Nguyen Huu Phat - phat.nguyenluxefats@gmail.com - https://github.com/NguyenHuuPhat2203
