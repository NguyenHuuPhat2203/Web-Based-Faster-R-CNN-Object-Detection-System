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

## Sync with Hugging Face

One Hugging Face model repo mirrors every weight file under `registry/`,
`checkpoints/`, and `.weights/` at the same paths. `models.lock.json` (repo root,
committed) pins the Hub commit that matches the code. Credentials and the repo id
come from the root `.env` (`HF_TOKEN`, `HF_MODEL_REPO_ID`; see `.env.example`).

```bash
python -m scripts.sync_models pull                   # restore the pinned weights
python -m scripts.sync_models push --dry-run         # list what would be uploaded
python -m scripts.sync_models push -m "faster_rcnn v1.1.0"
```

`push` uploads changed files in one commit and rewrites `models.lock.json`; commit
that file together with the new `metadata.json`. The repo is `--repo-id`, else
`HF_MODEL_REPO_ID`, else the one in `models.lock.json`; a missing repo is created
as private. Nothing is deleted on the Hub.

After a manual promote (below), run `push` so other machines can `pull` it.

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

The backend image is **thin** (no baked weights). Pull, then mount host artifacts:

```bash
python -m scripts.sync_models pull
docker compose up --build
```

Volumes (see `docker-compose.yml`):

- `./registry` mounted at `/registry` (`REGISTRY_DIR`)
- `./checkpoints` mounted at `/checkpoints` (`CHECKPOINT_DIR`)
- `./.weights` mounted at `/weights` (`SWIN_UNETR_WEIGHTS`)

At runtime the API prefers registry safetensors, then the local `checkpoints/model.pth`.
