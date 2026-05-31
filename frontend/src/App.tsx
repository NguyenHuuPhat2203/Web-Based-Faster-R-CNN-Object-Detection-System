import { useState, useRef, useEffect, useCallback } from "react";
import {
  Upload, AlertCircle, Download, Sliders, Clock,
  Image as ImageIcon, RotateCcw, Camera, Cpu, FileText,
  Thermometer, Ruler, Columns, Eye, EyeOff,
} from "lucide-react";
import { useAuth } from "./context/AuthContext";
import { Login } from "./components/Login";
import { Register } from "./components/Register";
import { Navbar } from "./components/Navbar";
import { OAuthCallback } from "./components/OAuthCallback";
import { WebcamCapture } from "./components/WebcamCapture";
import {
  predict,
  listImages,
  getImageUrl,
  getReportUrl,
  authFetch,
  type PredictionResult,
  type ImageInfo,
  type ModelType,
} from "./lib/api";

/* ---------- Types ---------- */

type Page = "login" | "register" | "oauth-callback" | "detector";
type ViewMode = "original" | "annotated" | "split";

interface Measurement {
  x1: number; y1: number; x2: number; y2: number;
}

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

/* ---------- App ---------- */

function App() {
  const { isAuth, loading } = useAuth();
  const [page, setPage] = useState<Page>(() => {
    const path = window.location.pathname;
    if (path.startsWith("/oauth-callback")) return "oauth-callback";
    return "login";
  });

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

  // -- View mode --
  const [viewMode, setViewMode] = useState<ViewMode>("annotated");
  const [showHeatmap, setShowHeatmap] = useState(false);

  // -- Measurement tool --
  const [measuring, setMeasuring] = useState(false);
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [measureStart, setMeasureStart] = useState<{ x: number; y: number } | null>(null);

  // -- Drag-and-drop --
  const [isDragging, setIsDragging] = useState(false);

  // -- History state --
  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState<ImageInfo[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const heatmapImgRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const splitCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Once authenticated, navigate to the detector
  useEffect(() => {
    if (isAuth && page !== "detector") {
      setPage("detector");
    }
  }, [isAuth, page]);

  /* ---------- File handling ---------- */

  const loadFile = useCallback((file: File) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResults(null);
    setDetectError("");
    setLatestImageId(null);
    setMeasurements([]);
    setShowHeatmap(false);
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

  /* ---------- Drawing helpers ---------- */

  const drawAnnotations = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    mode: "detections" | "measurements" | "all",
  ) => {
    // Draw detection boxes
    if (results && (mode === "detections" || mode === "all")) {
      ctx.lineWidth = 4;
      ctx.font = "bold 22px Inter, system-ui, sans-serif";

      results.boxes.forEach((box, i) => {
        const [xmin, ymin, xmax, ymax] = box;
        const score = results.scores[i];
        const label = results.label_names?.[i] || LABEL_NAMES[results.labels[i]] || `Class ${results.labels[i]}`;

        ctx.strokeStyle = "#10b981";
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        const labelText = `${label}: ${(score * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = "#10b981";
        ctx.fillRect(xmin, ymin - 34, textWidth + 14, 34);
        ctx.fillStyle = "white";
        ctx.fillText(labelText, xmin + 7, ymin - 9);
      });
    }

    // Draw measurement lines
    if (mode === "measurements" || mode === "all") {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.font = "bold 14px Inter, system-ui, sans-serif";

      for (const m of measurements) {
        ctx.beginPath();
        ctx.moveTo(m.x1, m.y1);
        ctx.lineTo(m.x2, m.y2);
        ctx.stroke();

        // Draw endpoints
        ctx.setLineDash([]);
        ctx.fillStyle = "#f59e0b";
        ctx.beginPath();
        ctx.arc(m.x1, m.y1, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(m.x2, m.y2, 5, 0, Math.PI * 2);
        ctx.fill();

        // Distance label
        const dist = Math.sqrt((m.x2 - m.x1) ** 2 + (m.y2 - m.y1) ** 2);
        const mx = (m.x1 + m.x2) / 2;
        const my = (m.y1 + m.y2) / 2;
        const label = `${dist.toFixed(0)}px`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
        ctx.fillRect(mx - tw / 2 - 4, my - 12, tw + 8, 20);
        ctx.fillStyle = "white";
        ctx.fillText(label, mx - tw / 2, my + 4);

        ctx.setLineDash([6, 4]);
      }
      ctx.setLineDash([]);

      // Draw in-progress measurement
      if (measureStart) {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        // Will be completed by mousemove; for now, partial line
      }
      ctx.setLineDash([]);
    }
  }, [results, measurements, measureStart]);

  // Full canvas render combining base image + annotations
  const renderCanvas = useCallback(() => {
    if (!canvasRef.current || !imgRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;
    if (!ctx) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    if (viewMode === "original") {
      // Just show the original image, no annotations
      ctx.drawImage(img, 0, 0);
      return;
    }

    ctx.drawImage(img, 0, 0);

    if (viewMode === "annotated" || viewMode === "split") {
      drawAnnotations(ctx, canvas.width, canvas.height, "all");
    }
  }, [viewMode, drawAnnotations]);

  useEffect(() => {
    if (results) renderCanvas();
  }, [results, renderCanvas, showHeatmap, viewMode, measurements, measureStart]);

  /* ---------- Heatmap overlay ---------- */

  useEffect(() => {
    if (!showHeatmap || !results?.heatmap || !canvasRef.current || !imgRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;
    if (!ctx) return;

    // Re-draw base image + heatmap on top
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    if (viewMode === "original") {
      ctx.drawImage(img, 0, 0);
      return;
    }

    ctx.drawImage(img, 0, 0);

    // Draw heatmap with 50% opacity
    const hmImg = new window.Image();
    hmImg.onload = () => {
      ctx.globalAlpha = 0.5;
      ctx.drawImage(hmImg, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
      // Re-draw annotations on top
      drawAnnotations(ctx, canvas.width, canvas.height, "all");
    };
    hmImg.src = `data:image/png;base64,${results.heatmap}`;
  }, [showHeatmap, results, viewMode, drawAnnotations]);

  /* ---------- Predict ---------- */

  const handleDetect = async () => {
    if (!image) return;
    setDetectError("");
    setLoadingPred(true);
    setLatestImageId(null);
    try {
      const data = await predict(image, threshold, model);
      setResults(data);
    } catch (err: any) {
      setDetectError(err.message || "Detection failed");
    } finally {
      setLoadingPred(false);
    }
  };

  /* ---------- Webcam ---------- */

  const handleWebcamResult = (result: PredictionResult) => {
    setResults(result);
  };
  const handleWebcamError = (errMsg: string) => {
    setDetectError(errMsg);
  };

  /* ---------- Download ---------- */

  const downloadAnnotated = () => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const link = document.createElement("a");
    link.download = `annotated-${image.name.replace(/\.[^.]+$/, "")}.png`;
    canvas.toBlob((blob) => {
      if (blob) { link.href = URL.createObjectURL(blob); link.click(); }
    });
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

  /* ---------- Measurement tool ---------- */

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!measuring || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (!measureStart) {
      setMeasureStart({ x, y });
    } else {
      setMeasurements(prev => [...prev, { x1: measureStart.x, y1: measureStart.y, x2: x, y2: y }]);
      setMeasureStart(null);
    }
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!measuring || !measureStart || !canvasRef.current) return;
    // Re-render to show live line preview — handled by useEffect on measureStart
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx || !imgRef.current) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Redraw
    canvas.width = imgRef.current.naturalWidth;
    canvas.height = imgRef.current.naturalHeight;
    ctx.drawImage(imgRef.current, 0, 0);

    drawAnnotations(ctx, canvas.width, canvas.height, "all");

    // Draw in-progress line
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(measureStart.x, measureStart.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  /* ---------- Reset ---------- */

  const reset = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    setDetectError("");
    setLatestImageId(null);
    setMeasurements([]);
    setMeasureStart(null);
    setShowHeatmap(false);
  };

  /* ---------- History ---------- */

  const fetchHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const items = await listImages();
      setHistory(items);
    } catch (err: any) {
      setHistoryError(err.message || "Failed to load history");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const toggleHistory = () => {
    const next = !showHistory;
    setShowHistory(next);
    if (next) fetchHistory();
  };

  /* ---------- Router ---------- */

  if (loading) {
    return <div className="loading-screen"><div className="spinner" /></div>;
  }

  if (page === "oauth-callback") {
    return <OAuthCallback />;
  }

  if (!isAuth) {
    if (page === "register") {
      return <Register onSwitchToLogin={() => setPage("login")} />;
    }
    return <Login onSwitchToRegister={() => setPage("register")} />;
  }

  /* ---------- Main detector page ---------- */

  return (
    <>
      <Navbar />
      <div className="container">
        <h1 className="title">Brain Tumor Detector</h1>

        {detectError && (
          <div className="auth-error" style={{ marginBottom: "1rem" }}>
            <AlertCircle size={18} />
            <span>{detectError}</span>
          </div>
        )}

        {/* ── Toolbar ── */}
        <div className="toolbar">
          <div className="toolbar-group">
            <Cpu size={16} />
            <select className="model-select" value={model}
              onChange={(e) => setModel(e.target.value as ModelType)}>
              {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
          <button className={`btn btn-ghost ${useWebcam ? "active" : ""}`}
            onClick={() => { setUseWebcam(!useWebcam); reset(); }}>
            <Camera size={16} /> {useWebcam ? "Upload Instead" : "Live Webcam"}
          </button>
        </div>

        <div className="glass-card">
          {useWebcam ? (
            <WebcamCapture model={model} threshold={threshold}
              onResult={handleWebcamResult} onError={handleWebcamError} />
          ) : !preview ? (
            <label className={`upload-zone${isDragging ? " dragover" : ""}`}
              onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
              <input ref={fileInputRef} type="file" hidden accept="image/*" onChange={handleFileChange} />
              <Upload size={48} className="mb-4" />
              <p>Click or drag image to upload</p>
              <p className="text-muted">Supports JPG, PNG &mdash; MRI scans preferred</p>
            </label>
          ) : (
            <div className="text-center">
              <div className="preview-container">
                <img ref={imgRef} src={preview} alt="Preview"
                  style={{ display: results && viewMode !== "original" ? "none" : "block", maxWidth: "100%" }}
                  onLoad={() => results && renderCanvas()} />
                <canvas ref={canvasRef}
                  style={{ display: results && viewMode !== "original" ? "block" : "none", cursor: measuring ? "crosshair" : "default" }}
                  onClick={handleCanvasClick} onMouseMove={handleCanvasMove} />
                {/* Hidden heatmap image for preloading */}
                {results?.heatmap && (
                  <img ref={heatmapImgRef} src={`data:image/png;base64,${results.heatmap}`}
                    style={{ display: "none" }} alt="" />
                )}
              </div>

              {/* ── Threshold slider ── */}
              <div className="threshold-row">
                <Sliders size={16} />
                <label className="threshold-label">
                  Confidence: <strong>{(threshold * 100).toFixed(0)}%</strong>
                </label>
                <input type="range" min="0.05" max="0.95" step="0.05" value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))} className="threshold-slider" />
              </div>

              {/* ── Action buttons ── */}
              <div className="action-row">
                <button className="btn btn-secondary" onClick={reset}><RotateCcw size={18} /> Reset</button>
                <button className="btn btn-primary" onClick={handleDetect} disabled={loadingPred}>
                  {loadingPred ? <><span className="spinner-small" /> Processing&hellip;</> : "Detect"}
                </button>
                {results && <button className="btn btn-accent" onClick={downloadAnnotated}><Download size={18} /> Download</button>}
                {results && <button className="btn btn-secondary" onClick={downloadReport}><FileText size={18} /> Report</button>}
              </div>

              {/* ── View mode + extras toolbar ── */}
              {results && (
                <div className="view-toolbar">
                  <div className="toolbar-group">
                    <Eye size={14} />
                    {(["annotated", "original", "split"] as ViewMode[]).map(mode => (
                      <button key={mode}
                        className={`btn btn-tiny ${viewMode === mode ? "active" : ""}`}
                        onClick={() => setViewMode(mode)}>
                        {mode === "annotated" ? "Annotated" : mode === "original" ? "Original" : "Side-by-side"}
                      </button>
                    ))}
                  </div>

                  {results.heatmap && (
                    <button className={`btn btn-tiny ${showHeatmap ? "active" : ""}`}
                      onClick={() => setShowHeatmap(!showHeatmap)}>
                      <Thermometer size={14} /> {showHeatmap ? "Hide Heatmap" : "Heatmap"}
                    </button>
                  )}

                  <button className={`btn btn-tiny ${measuring ? "active" : ""}`}
                    onClick={() => { setMeasuring(!measuring); if (!measuring) setMeasureStart(null); }}>
                    <Ruler size={14} /> {measuring ? "Done Measuring" : "Measure"}
                  </button>

                  {measurements.length > 0 && (
                    <button className="btn btn-tiny" onClick={() => { setMeasurements([]); setMeasureStart(null); }}>
                      Clear Lines
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── Results section ── */}
        {results && !useWebcam && results.boxes.length > 0 && (
          <div className="stats">
            <div className="stat-card"><h3>Detections</h3><p className="stat-value">{results.boxes.length}</p></div>
            <div className="stat-card"><h3>Avg Score</h3><p className="stat-value">
              {((results.scores.reduce((a, b) => a + b, 0) / results.scores.length) * 100).toFixed(1)}%
            </p></div>
            {results.label_names?.length > 0 && (
              <div className="stat-card"><h3>Top Class</h3><p className="stat-value-sm">{results.label_names[0]}</p></div>
            )}
          </div>
        )}

        {results && !useWebcam && results.boxes.length === 0 && (
          <div className="no-detections">
            <AlertCircle size={24} />
            <p><strong>No tumors detected</strong> above {(threshold * 100).toFixed(0)}% confidence.</p>
            <p className="text-muted">Try lowering the threshold or uploading a different scan.</p>
          </div>
        )}

        {/* ── History ── */}
        <div className="history-section">
          <button className="history-toggle" onClick={toggleHistory}>
            <Clock size={18} /> {showHistory ? "Hide History" : "View History"}
          </button>
          {showHistory && (
            <div className="history-content">
              {historyLoading && <div className="history-status"><div className="spinner" /><p>Loading history&hellip;</p></div>}
              {historyError && (
                <div className="history-status">
                  <AlertCircle size={18} /><p>{historyError}</p>
                  <button className="btn btn-secondary" onClick={fetchHistory} style={{ width: "auto", marginTop: "0.5rem" }}>Retry</button>
                </div>
              )}
              {!historyLoading && !historyError && history.length === 0 && (
                <div className="history-empty"><ImageIcon size={32} /><p>No images yet. Upload an MRI scan to get started.</p></div>
              )}
              {!historyLoading && !historyError && history.length > 0 && (
                <div className="history-grid">
                  {history.map(item => (
                    <div key={item.id} className="history-card"
                      onClick={() => {
                        setPreview(getImageUrl(item.id));
                        if (item.detection_result) setResults(item.detection_result as PredictionResult);
                        setLatestImageId(item.id);
                        setImage(null);
                      }}>
                      <img src={getImageUrl(item.id)} alt={item.original_name} className="history-thumb" loading="lazy" />
                      <div className="history-info">
                        <p className="history-name" title={item.original_name}>{item.original_name}</p>
                        <p className="history-meta">
                          {item.detection_result ? `${item.detection_result.boxes.length} detections` : "No results"}
                          &nbsp;&middot;&nbsp;{new Date(item.uploaded_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;
