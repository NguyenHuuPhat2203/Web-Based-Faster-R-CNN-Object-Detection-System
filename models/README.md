# Archived training notebooks

Production training lives under [`training/`](../training/).

These notebooks are kept for historical reference / Colab experiments.
Do **not** import `app.*` from notebooks. Extract stable code into `ml/` or `training/`.

| Notebook | Notes |
|----------|-------|
| `train.ipynb` | Legacy Faster R-CNN training |
| `train_yolov8.ipynb` | Legacy YOLO training |
| `faster_rcnn_scratch.ipynb` | Scratch experiments |

## Preferred CLI

```bash
# Local E2E smoke (no registry overwrite)
python training/scripts/run_e2e_smoke.py

# Full dataset + train
python training/scripts/download_dataset.py
python training/scripts/train_faster_rcnn.py
python training/scripts/train_yolo.py

# Promote with an explicit version
python training/scripts/promote_checkpoint.py --model faster_rcnn \
  --checkpoint training/outputs/faster_rcnn/best_model.pth --version v1.1.0

# Or promote from Hugging Face for serve
python training/scripts/promote_from_hf.py
python training/scripts/verify_dataset.py --dataset-dir data/raw/v1 --write
```
