#!/usr/bin/env bash
# Download the trained model checkpoint.
# Usage: bash download_model.sh

set -euo pipefail

MODEL_DIR="checkpoints"
MODEL_PATH="${MODEL_DIR}/model.pth"

# ── Option 1: Hugging Face (recommended) ─────────────────────────
# HF_REPO="your-org/brain-tumor-faster-rcnn"
# HF_FILE="model.pth"
# echo "Downloading from Hugging Face..."
# curl -L "https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILE}" -o "${MODEL_PATH}"

# ── Option 2: Google Drive ────────────────────────────────────────
# FILE_ID="your-google-drive-file-id"
# echo "Downloading from Google Drive..."
# curl -sc /tmp/gd-cookie "https://docs.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
# CODE="$(awk '/_warning_/ {print $NF}' /tmp/gd-cookie)"
# curl -Lb /tmp/gd-cookie "https://docs.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o "${MODEL_PATH}"

# ── Option 3: Manual placeholder ──────────────────────────────────
if [ ! -f "${MODEL_PATH}" ]; then
    echo "No automatic download configured."
    echo "Place your trained model.pth in ${MODEL_DIR}/"
    echo "See README.md for training instructions."
    mkdir -p "${MODEL_DIR}"
    touch "${MODEL_DIR}/.gitkeep"
fi
