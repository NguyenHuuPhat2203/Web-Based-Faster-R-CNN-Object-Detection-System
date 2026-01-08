import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import numpy as np

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# DEVICE = "device"
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cpu')
NUM_CLASSES = 3

def get_model(num_classes):
    # Load a model pre-trained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(NUM_CLASSES)
checkpoint_path = "../checkpoints/model.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
# 
model.to(DEVICE)
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Preprocess
    img_tensor = torchvision.transforms.ToTensor()(image).to(DEVICE)

    # Inference
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]

    # Filter by threshold
    mask = pred['scores'] > threshold
    boxes = pred['boxes'][mask].cpu().numpy().tolist()
    labels = pred['labels'][mask].cpu().numpy().tolist()
    scores = pred['scores'][mask].cpu().numpy().tolist()

    return {
        "boxes": boxes, # [xmin, ymin, xmax, ymax]
        "labels": labels,
        "scores": scores
    }

@app.get("/")
def read_root():
    return {"status": "Backend is running", "device": str(DEVICE)}
