# Close Detector → History → Report

Date: 2026-09-19

## Goal

Make the existing 2D loop work end-to-end: detect → stored scan id → PDF with boxes → History that can filter, sort, and re-run. No new screens, no 3D, no heatmap persistence.

## Why it is broken today

- `ScanPersistenceService.save_scan_result` already returns `Image.id`, but `PredictionService.run` drops it. `PredictionResult` has no `image_id`.
- Detector keeps `latestImageId` and never sets it, so the Report button is a no-op.
- History PDF works (it already has the id). The PDF titled "Annotated Image" embeds the raw file and omits `model_type` / `model_version` / `threshold`.
- History search filters filename client-side. Filter and Sort buttons do nothing. No re-run.

## Non-goals

Case Viewer, dual-model side-by-side, PACS window/level, mm measurements, heatmap storage, Dashboard fake charts, API pagination.

## Contract

### 1. Persist id on the predict response

- Add optional `image_id: int | None = None` to `backend/schemas/prediction.py` `PredictionResult` (v1 body and v2 `result`).
- `PredictionService.run` uses the `Image` returned by `save_scan_result` and sets `image_id=image.id`.
- Routers stay unchanged; they already return `PredictionResult`.

Tests:

- `backend/tests/unit/test_prediction_service.py`: after `run()`, `result.image_id == saved.id`.
- `backend/tests/integration/test_predict.py` and `test_predict_v2.py`: response `image_id` matches `GET /images` first row.

### 2. Detector uses that id

- `frontend/src/lib/api.ts` `PredictionResult` gets `image_id?: number | null`.
- Detector `handleDetect`: `setLatestImageId(data.image_id ?? null)`.
- Webcam `onResult`: same.
- Report button disabled until `latestImageId` is set.
- Replace fake "Patient ID / T2-FLAIR / 128 / 1.0mm" with scan id, filename, model, version. Drop "Temporal Lobe" and fake `42ms`.

### 3. PDF matches stored detections

- `ReportService` draws boxes + label + score on a RGB copy of the stored image (Pillow), then embeds that PNG. Heading stays "Annotated Image".
- Meta table: Scan ID `IMG-{id:06d}`, file, uploaded, generated, model, version, threshold, detection count, avg confidence. Keep the existing medical disclaimer.
- No Grad-CAM in the PDF.

Tests:

- Unit: given a tiny PNG + one box, annotated bytes differ from the original and a pixel on the box outline is not the background color.
- Existing `test_report.py` still gets `%PDF`.

### 4. History filter / sort / re-run (client)

`GET /images` already returns the full list with `model_type`, `model_version`, `threshold`, detections. No new list API.

Extract `frontend/src/lib/historyList.ts`:

- search: filename contains (case-insensitive)
- model: `all` | `faster-rcnn` | `yolov8`
- detections: `all` | `with` | `without`
- sort: `newest` | `oldest` by `uploaded_at`

Wire the existing Filter / Sort buttons to those controls. Show `model_type` + version on each row.

Re-run: `GET /images/{id}` blob → `predict(file, img.threshold ?? 0.5, otherModel)`. Persistence already inserts a new row. Refresh the list. Other model = YOLO if current is Faster R-CNN, else Faster R-CNN.

Tests: vitest on `historyList.ts`. No component snapshot suite.

## Order of work

1. Failing unit + integration tests for `image_id`, then schema/service.
2. Detector + `api.ts` wiring.
3. Failing annotated-preview unit test, then `ReportService`.
4. `historyList.ts` tests, then History UI + re-run.

## Verify

```
./backend/.venv/Scripts/python.exe -m pytest backend/tests/unit/test_prediction_service.py backend/tests/unit/test_report_service.py backend/tests/integration/test_predict.py backend/tests/integration/test_predict_v2.py backend/tests/integration/test_predict_yolo.py backend/tests/integration/test_report.py backend/tests/integration/test_images.py -q
```

```
npm test   # in frontend/
```

## Out of this change

Formatters, linters, git, Case Viewer, heatmap persistence.
