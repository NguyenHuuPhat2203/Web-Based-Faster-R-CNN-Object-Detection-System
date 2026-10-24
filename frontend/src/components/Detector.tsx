import React, { useEffect, useRef, useState } from "react";
import {
  Upload, AlertCircle, Sliders, RotateCcw, Camera, Cpu, FileText,
  CheckCircle, Brain, Activity, Server,
} from "lucide-react";
import { WebcamCapture } from "./WebcamCapture";
import { ScanViewer } from "./ScanViewer";
import { ConfidenceSlider } from "./ConfidenceSlider";
import { messageOf, downloadReport, predict, type PredictionResult, type ModelType } from "../lib/api";
import { DEFAULT_CONFIDENCE, displayLabel, isTumor, visibleDetections } from "../lib/detections";

/* ---------- Model options ---------- */

const MODELS: { value: ModelType; label: string }[] = [
  { value: "faster-rcnn", label: "Faster R-CNN" },
  { value: "yolov8", label: "YOLOv8" },
];

export function Detector() {
  // -- Mode --
  const [useWebcam, setUseWebcam] = useState(false);

  // -- Detection state --
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loadingPred, setLoadingPred] = useState(false);
  const [results, setResults] = useState<PredictionResult | null>(null);
  const [detectError, setDetectError] = useState("");
  const [threshold, setThreshold] = useState(DEFAULT_CONFIDENCE);
  const [model, setModel] = useState<ModelType>("faster-rcnn");
  const [latestImageId, setLatestImageId] = useState<number | null>(null);
  const [imageFileName, setImageFileName] = useState<string>("");

  // -- Drag-and-drop --
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Identifies the latest detect request so a stale response cannot overwrite a newer scan.
  const runId = useRef(0);

  // Object URL lives exactly as long as the selected file.
  useEffect(() => {
    if (!image) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(image);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [image]);

  /* ---------- File handling ---------- */

  const clearOutcome = () => {
    runId.current += 1;
    setLoadingPred(false);
    setResults(null);
    setDetectError("");
    setLatestImageId(null);
  };

  const loadFile = (file: File) => {
    clearOutcome();
    setImage(file);
    setImageFileName(file.name);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadFile(file);
    e.target.value = ""; // allow re-selecting the same file after Reset
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) loadFile(file);
  };


  /* ---------- Predict ---------- */

  const handleDetect = async () => {
    if (!image) return;
    const id = ++runId.current;
    setDetectError("");
    setLoadingPred(true);
    setLatestImageId(null);
    try {
      const data = await predict(image, threshold, model, {
        explain: model === "faster-rcnn",
      });
      if (id !== runId.current) return;
      setResults(data);
      setLatestImageId(data.image_id ?? null);
    } catch (err) {
      if (id === runId.current) setDetectError(messageOf(err, "Detection failed"));
    } finally {
      if (id === runId.current) setLoadingPred(false);
    }
  };

  /* ---------- Report ---------- */

  const handleDownloadReport = async () => {
    if (!latestImageId) return;
    try {
      await downloadReport(latestImageId);
    } catch (err) {
      setDetectError(messageOf(err, "Failed to generate report"));
    }
  };

  /* ---------- Reset ---------- */

  const reset = () => {
    clearOutcome();
    setImage(null);
    setImageFileName("");
  };

  /* ─── Computed values for sidebar ─── */
  const visible = results ? visibleDetections(results, threshold) : [];
  const tumors = visible.filter((d) => isTumor(d.labelName));
  const avgConfidence =
    visible.length > 0
      ? ((visible.reduce((sum, d) => sum + d.score, 0) / visible.length) * 100).toFixed(1)
      : null;
  const topDetection = visible.reduce<(typeof visible)[number] | null>(
    (best, d) => (!best || d.score > best.score ? d : best),
    null,
  );
  const topLabel = topDetection ? displayLabel(topDetection.labelName) : null;

  const thresholdRow = (
    <div className="threshold-row">
      <Sliders size={16} />
      <span className="threshold-label">
        Confidence: <strong>{(threshold * 100).toFixed(0)}%</strong>
      </span>
      <ConfidenceSlider value={threshold} onChange={setThreshold} />
    </div>
  );

  return (
    <div className="page-container">
      <h1 className="page-title">Brain Tumor Analysis</h1>

      {detectError && (
        <div className="auth-error" style={{ marginBottom: "1rem" }}>
          <AlertCircle size={18} />
          <span>{detectError}</span>
        </div>
      )}

      <div className="detector-layout">
        {/* ─── Main viewer ─── */}
        <div className="detector-main">
          <div className="glass-card">
            {useWebcam ? (
              <>
                <WebcamCapture model={model} threshold={threshold}
                  onResult={(r) => {
                    setResults(r);
                    setLatestImageId(r.image_id ?? null);
                  }} onError={(e) => setDetectError(e)} />
                {thresholdRow}
              </>
            ) : !preview ? (
              <label className={`upload-zone${isDragging ? " dragover" : ""}`}
                onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
                <input ref={fileInputRef} type="file" hidden accept="image/*" onChange={handleFileChange} />
                <Upload size={48} className="mb-4" />
                <p>Drop an MRI image (JPEG or PNG) here</p>
                <p className="text-muted" style={{ fontSize: "0.85rem", marginTop: "0.3rem" }}>
                  Supports T1, T2, and FLAIR weighted sequences
                </p>
              </label>
            ) : (
              <div className="text-center">
                <ScanViewer imageUrl={preview} results={results} threshold={threshold} />

                {thresholdRow}

                {/* Action buttons */}
                <div className="action-row">
                  <button className="btn btn-secondary" onClick={reset}><RotateCcw size={18} /> Reset</button>
                  <button className="btn btn-primary" onClick={handleDetect} disabled={loadingPred}>
                    {loadingPred ? <><span className="spinner-small" /> Processing&hellip;</> : "Detect"}
                  </button>
                  {results && (
                    <button
                      className="btn btn-secondary"
                      onClick={handleDownloadReport}
                      disabled={!latestImageId}
                    >
                      <FileText size={18} /> Report
                    </button>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Results stats below viewer */}
          {results && !useWebcam && visible.length > 0 && (
            <div className="stats">
              <div className="stat-card"><h3>Detections</h3><p className="stat-value">{visible.length}</p></div>
              <div className="stat-card"><h3>Avg Score</h3><p className="stat-value">{avgConfidence}%</p></div>
              {topLabel && (
                <div className="stat-card"><h3>Top Class</h3><p className="stat-value-sm">{topLabel}</p></div>
              )}
            </div>
          )}

          {results && !useWebcam && tumors.length === 0 && (
            <div className="no-detections">
              <AlertCircle size={24} />
              <p><strong>No tumors detected</strong> above {(threshold * 100).toFixed(0)}% confidence.</p>
              <p className="text-muted">Try lowering the threshold or uploading a different scan.</p>
            </div>
          )}
        </div>

        {/* ─── Right control panel ─── */}
        <div className="detector-sidebar">
          {/* Detection metrics */}
          <div className="metric-card">
            <div className="metric-card-icon green">
              <CheckCircle size={20} />
            </div>
            <div className="metric-card-body">
              <div className="metric-card-value">{results ? visible.length : "—"}</div>
              <div className="metric-card-label">Detections Found</div>
              {tumors.length > 0 && (
                <div className="metric-card-label-sub">
                  <CheckCircle size={10} />
                  {tumors.length} tumor region{tumors.length === 1 ? "" : "s"}
                </div>
              )}
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-card-icon cyan">
              <Activity size={20} />
            </div>
            <div className="metric-card-body">
              <div className="metric-card-value">{avgConfidence ? `${avgConfidence}%` : "—"}</div>
              <div className="metric-card-label">Avg Confidence</div>
              {topLabel && (
                <div className="metric-card-label-sub">
                  <Brain size={10} />
                  Primary: {topLabel}
                </div>
              )}
            </div>
          </div>

          {topLabel && (
            <div className="metric-card">
              <div className="metric-card-icon amber">
                <Brain size={20} />
              </div>
              <div className="metric-card-body">
                <div className="metric-card-value" style={{ fontSize: "1rem" }}>{topLabel}</div>
                <div className="metric-card-label">Primary Classification</div>
              </div>
            </div>
          )}

          <div className="metadata-card">
            <div className="metadata-title">
              <Server size={14} />
              Scan
            </div>
            <div className="metadata-grid">
              <div className="metadata-item">
                <span className="metadata-key">Scan ID</span>
                <span className="metadata-value">
                  {latestImageId ? `IMG-${String(latestImageId).padStart(6, "0")}` : "—"}
                </span>
              </div>
              <div className="metadata-item">
                <span className="metadata-key">File</span>
                <span className="metadata-value">{imageFileName || "—"}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-key">Model</span>
                <span className="metadata-value">{results?.model_type || model}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-key">Version</span>
                <span className="metadata-value">{results?.model_version || "—"}</span>
              </div>
            </div>
          </div>

          {/* Model & Device */}
          <div className="control-group">
            <div className="control-group-title">
              <Cpu size={14} />
              Model &amp; Hardware
            </div>
            <div className="control-row">
              <span className="control-label">Architecture</span>
              <select
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "6px",
                  color: "var(--text)",
                  padding: "0.3rem 0.6rem",
                  fontSize: "0.78rem",
                  fontFamily: "inherit",
                  cursor: "pointer",
                  outline: "none",
                  maxWidth: "140px",
                }}
                value={model}
                onChange={(e) => setModel(e.target.value as ModelType)}
              >
                {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
            </div>
            <div className="control-row">
              <span className="control-label">
                <Camera size={12} />
                Input Mode
              </span>
              <button
                className={`btn btn-tiny ${useWebcam ? "active" : ""}`}
                onClick={() => { setUseWebcam(!useWebcam); reset(); }}
                style={{ fontSize: "0.72rem" }}
              >
                {useWebcam ? "Upload" : "Webcam"}
              </button>
            </div>
            {loadingPred && (
              <div className="control-row">
                <span className="control-label">
                  <Activity size={12} />
                  Status
                </span>
                <span className="control-value" style={{ fontSize: "0.75rem" }}>
                  processing…
                </span>
              </div>
            )}
          </div>

          {!preview && !useWebcam && (
            <button
              className="btn btn-primary"
              onClick={() => fileInputRef.current?.click()}
              style={{ marginTop: "auto" }}
            >
              <Upload size={16} /> Upload New Scan
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
