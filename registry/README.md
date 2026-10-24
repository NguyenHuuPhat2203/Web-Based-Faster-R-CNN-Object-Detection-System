# Model Registry

Promoted artifacts live here. Do not deploy directly from `checkpoints/` or `training/outputs/`.

Weights (`.safetensors`, `.pt`, `.pth`) are **gitignored**. Commit `metadata.json` only.
`metadata.json` records `weights_sha256`; `ml/models/yolo.py` verifies it before loading YOLO weights.

## Layout

```
registry/
|-- faster_rcnn/<version>/
|   |-- model.safetensors   # gitignored
|   `-- metadata.json
`-- yolo/<version>/
    |-- yolov8_tumor.pt     # gitignored
    `-- metadata.json
```

`<version>` is `vMAJOR.MINOR[.PATCH][-build]`. `v1.0.0` is the production slot and
needs `--i-know`; use another version (for example `v0.0.0-smoke`) for experiments.

## One-shot promote from Hugging Face

Downloads the Faster R-CNN `model.pth` into `checkpoints/`, converts it to safetensors,
and writes registry metadata. Promotes YOLO only if `checkpoints/yolov8_tumor.pt`
already exists. `--revision` (commit SHA or tag) is required.

```bash
python -m training.scripts.promote_from_hf --revision <commit-sha> --i-know
# or reuse a local pth:
python -m training.scripts.promote_from_hf --revision <commit-sha> --skip-download --i-know
```

## Manual promote

```bash
python -m training.scripts.export_safetensors \
  --checkpoint checkpoints/model.pth \
  --version v1.0.0 --i-know \
  --dataset-version v1 \
  --dataset-hash <sha256> \
  --map50 0.69 --map5095 0.49

python -m training.scripts.promote_yolo \
  --checkpoint checkpoints/yolov8_tumor.pt \
  --version v1.0.0 --i-know \
  --dataset-version v1 \
  --dataset-hash <sha256>
```

`export_safetensors` refuses checkpoints whose keys or shapes do not match the
3-class Faster R-CNN served by the backend.

## Docker / Compose

The backend image is **thin** (no baked weights). Mount host artifacts:

```bash
docker compose up --build
```

Volumes (see `docker-compose.yml`):

- `./registry` mounted at `/registry` (`REGISTRY_DIR`)
- `./checkpoints` mounted at `/checkpoints` (`CHECKPOINT_DIR`)

At runtime the API prefers registry safetensors, then local checkpoints, then Hugging Face download fallback.
