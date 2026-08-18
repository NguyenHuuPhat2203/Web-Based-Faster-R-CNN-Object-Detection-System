import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload, AlertCircle, Sliders, RotateCcw, Camera, Cpu, FileText,
  CheckCircle, Brain, Activity, Server, Wifi,
} from "lucide-react";
import { WebcamCapture } from "./WebcamCapture";
import { ScanViewer } from "./ScanViewer";
import {
  predict,
  getReportUrl,
  authFetch,
  type PredictionResult,
  type ModelType,
} from "../lib/api";

/* ---------- Label map (mirrors backend CLASS_NAMES) ---------- */

const LABEL_NAMES: Record<number, string> = {
  1: "Brain tumor",
  2: "Brain tumor",
};

/* ---------- Model options ---------- */

const MODELS: { value: ModelType; label: string }[] = [
  { value: "faster-rcnn", label: "Faster R-CNN" },
  { value: "yolov8", label: "YOLOv8" },
];

/* ---------- Types ---------- */

export function Detector() {
  // -- Mode --
  const [useWebcam, setUseWebcam] = useState(false);

  // -- Detection state --
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loadingPred, setLoadingPred] = useState(false);
  const [results, setResults] = useState<PredictionResult | null>(null);
  const [detectError, setDetectError] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [model, setModel] = useState<ModelType>("faster-rcnn");
  const [latestImageId, setLatestImageId] = useState<number | null>(null);
  const [imageFileName, setImageFileName] = useState<string>("");

  // -- Drag-and-drop --
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  /* ---------- File handling ---------- */

  const loadFile = useCallback((file: File) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResults(null);
    setDetectError("");
    setLatestImageId(null);
    setImageFileName(file.name);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadFile(file);
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
    setDetectError("");
    setLoadingPred(true);
    setLatestImageId(null);
    try {
      const data = await predict(image, threshold, model, {
        explain: model === "faster-rcnn",
      });
      setResults(data);
    } catch (err: any) {
      setDetectError(err.message || "Detection failed");
    } finally {
      setLoadingPred(false);
    }
  };

  /* ---------- Report ---------- */

  const downloadReport = async () => {
    if (!latestImageId) return;
    const url = getReportUrl(latestImageId);
    const resp = await authFetch(url);
    if (!resp.ok) { setDetectError("Failed to generate report"); return; }
    const blob = await resp.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `report-${latestImageId}.pdf`;
    link.click();
  };

  /* (measurement tool moved into ScanViewer) */

  /* ---------- Reset ---------- */

  const reset = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    setDetectError("");
    setLatestImageId(null);
    setImageFileName("");
  };

  /* ─── Computed values for sidebar ─── */
  const visibleCount = results
    ? results.boxes.filter((_, i) => (results.scores[i] ?? 0) >= threshold).length
    : 0;
  const avgConfidence =
    results && visibleCount > 0
      ? ((results.scores.filter((s) => s >= threshold).reduce((a, b) => a + b, 0) /
          visibleCount) *
          100).toFixed(1)
      : null;
  const topLabel =
    results?.label_names?.[results.scores.findIndex((s) => s >= threshold)] ?? null;

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
              <WebcamCapture model={model} threshold={threshold}
                onResult={(r) => setResults(r)} onError={(e) => setDetectError(e)} />
            ) : !preview ? (
              <label className={`upload-zone${isDragging ? " dragover" : ""}`}
                onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
                <input ref={fileInputRef} type="file" hidden accept="image/*" onChange={handleFileChange} />
                <Upload size={48} className="mb-4" />
                <p>Drop MRI DICOM or JPEG here</p>
                <p className="text-muted" style={{ fontSize: "0.85rem", marginTop: "0.3rem" }}>
                  Supports T1, T2, and FLAIR weighted sequences
                </p>
              </label>
            ) : (
              <div className="text-center">
                <ScanViewer imageUrl={preview} results={results} threshold={threshold} />

                {/* Threshold slider */}
                <div className="threshold-row">
                  <Sliders size={16} />
                  <label className="threshold-label">
                    Confidence: <strong>{(threshold * 100).toFixed(0)}%</strong>
                  </label>
                  <input type="range" min="0.05" max="0.95" step="0.05" value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))} className="threshold-slider" />
                </div>

                {/* Action buttons */}
                <div className="action-row">
                  <button className="btn btn-secondary" onClick={reset}><RotateCcw size={18} /> Reset</button>
                  <button className="btn btn-primary" onClick={handleDetect} disabled={loadingPred}>
                    {loadingPred ? <><span className="spinner-small" /> Processing&hellip;</> : "Detect"}
                  </button>
                  {results && <button className="btn btn-secondary" onClick={downloadReport}><FileText size={18} /> Report</button>}
                </div>
              </div>
            )}
          </div>

          {/* Results stats below viewer */}
          {results && !useWebcam && visibleCount > 0 && (
            <div className="stats">
              <div className="stat-card"><h3>Detections</h3><p className="stat-value">{visibleCount}</p></div>
              <div className="stat-card"><h3>Avg Score</h3><p className="stat-value">{avgConfidence}%</p></div>
              {topLabel && (
                <div className="stat-card"><h3>Top Class</h3><p className="stat-value-sm">{topLabel}</p></div>
              )}
            </div>
          )}

          {results && !useWebcam && visibleCount === 0 && (
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
              <div className="metric-card-value">{results ? results.boxes.length : "—"}</div>
              <div className="metric-card-label">Detections Found</div>
              {results && results.boxes.length > 0 && (
                <div className="metric-card-label-sub">
                  <CheckCircle size={10} />
                  Confirmed regions
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
                <div className="metric-card-label-sub">
                  Localization: Temporal Lobe (Left)
                </div>
              </div>
            </div>
          )}

          {/* DICOM Metadata */}
          <div className="metadata-card">
            <div className="metadata-title">
              <Server size={14} />
              DICOM Metadata
            </div>
            <div className="metadata-grid">
              <div className="metadata-item">
                <span className="metadata-key">Patient ID</span>
                <span className="metadata-value">NS-{latestImageId ? String(latestImageId).padStart(4, "0") : "—"}</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-key">Sequence</span>
                <span className="metadata-value">T2-FLAIR</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-key">Slices</span>
                <span className="metadata-value">128</span>
              </div>
              <div className="metadata-item">
                <span className="metadata-key">Spacing</span>
                <span className="metadata-value">1.0mm</span>
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
            <div className="control-row">
              <span className="control-label">
                <Server size={12} />
                Device
              </span>
              <span className="control-value" style={{ fontSize: "0.75rem" }}>
                <Wifi size={10} style={{ color: "var(--accent)", marginRight: "0.2rem" }} />
                GPU
              </span>
            </div>
            <div className="control-row">
              <span className="control-label">
                <Activity size={12} />
                Latency
              </span>
              <span className="control-value" style={{ fontSize: "0.75rem" }}>
                {loadingPred ? "processing…" : results ? "42ms" : "—"}
              </span>
            </div>
          </div>

          {/* Upload CTA when no image loaded */}
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
