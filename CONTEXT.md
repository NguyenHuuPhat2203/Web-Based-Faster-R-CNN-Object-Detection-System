# Context — Brain Tumor Detector

## Scan
An uploaded MRI image plus its detections, owned by a user.

## Detection
One predicted region: a bounding box (xyxy), a class label, and a confidence score.

## Inference
Running a detection model (Faster R-CNN or YOLO) on a scan to produce detections.

## Model adapter
The per-model piece behind the inference seam: preprocessing, forward pass,
and optional explainability. Two adapters exist: Faster R-CNN and YOLO.

## Case
A BraTS subject: a directory holding four MRI modalities (flair, t1, t1ce, t2)
plus an expert segmentation. Read-only access goes through the **case store**
module — no other code knows the on-disk layout.

## Region
A tumor sub-region (NCR, ED, ET) with a fixed label id, name, and display
color. The single vocabulary lives in ``ml.core.labels``; model decode, case
metrics, and the viewer legend all derive from it.

## Score floor
The minimum confidence score the server returns; the client slider filters above it.

## Explainability
A Grad-CAM heatmap overlaid on a scan, available only for Faster R-CNN.
