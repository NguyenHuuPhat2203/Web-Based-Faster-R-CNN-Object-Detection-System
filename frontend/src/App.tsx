import React, { useState, useRef, useEffect } from 'react';
import { Upload, Image as ImageIcon, CheckCircle, AlertCircle } from 'lucide-react';

const APP_URL = "http://localhost:8000";

/* ---------- Types ---------- */
interface DetectionResult {
  boxes: number[][];
  scores: number[];
  labels: number[];
}

function App() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [results, setResults] = useState<DetectionResult | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  /* ---------- Handlers ---------- */
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResults(null);
  };

  const drawDetections = () => {
    if (!results || !canvasRef.current || !imgRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;
    if (!ctx) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    ctx.drawImage(img, 0, 0);
    ctx.lineWidth = 4;
    ctx.font = "24px Inter";

    results.boxes.forEach((box, i) => {
      const [xmin, ymin, xmax, ymax] = box;
      const score = results.scores[i];

      ctx.strokeStyle = "#10b981";
      ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

      const labelText = `Score: ${(score * 100).toFixed(1)}%`;
      const textWidth = ctx.measureText(labelText).width;

      ctx.fillStyle = "#10b981";
      ctx.fillRect(xmin, ymin - 30, textWidth + 10, 30);

      ctx.fillStyle = "white";
      ctx.fillText(labelText, xmin + 5, ymin - 7);
    });
  };

  useEffect(() => {
    if (results) drawDetections();
  }, [results]);

  const handleUpload = async () => {
    if (!image) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch(`${APP_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data: DetectionResult = await res.json();
      setResults(data);
    } catch (err) {
      console.error("Upload error:", err);
      alert("Failed to connect to backend");
    } finally {
      setLoading(false);
    }
  };

  /* ---------- UI ---------- */
  return (
    <div className="container">
      <h1 className="title">Brain Tumor Detector</h1>

      <div className="glass-card">
        {!preview ? (
          <label className="upload-zone">
            <input
              type="file"
              hidden
              accept="image/*"
              onChange={handleFileChange}
            />
            <Upload size={48} className="mb-4" />
            <p>Click or drag image to upload</p>
            <p className="text-muted">Supports JPG, PNG</p>
          </label>
        ) : (
          <div className="text-center">
            <div className="preview-container">
              <img
                ref={imgRef}
                src={preview}
                alt="Preview"
                style={{
                  display: results ? "none" : "block",
                  maxWidth: "100%",
                }}
                onLoad={() => results && drawDetections()}
              />
              <canvas
                ref={canvasRef}
                style={{ display: results ? "block" : "none" }}
              />
            </div>

            <div
              style={{
                marginTop: "2rem",
                display: "flex",
                gap: "1rem",
                justifyContent: "center",
              }}
            >
              <button
                className="btn"
                style={{ background: "#334155" }}
                onClick={() => {
                  setImage(null);
                  setPreview(null);
                  setResults(null);
                }}
              >
                Reset
              </button>

              <button
                className="btn"
                onClick={handleUpload}
                disabled={loading}
              >
                {loading ? "Processing..." : "Detect"}
              </button>
            </div>
          </div>
        )}
      </div>

      {results && (
        <div className="stats">
          <div className="stat-card">
            <h3>Detections</h3>
            <p className="stat-value">{results.boxes.length}</p>
          </div>

          <div className="stat-card">
            <h3>Avg Score</h3>
            <p className="stat-value">
              {(
                (results.scores.reduce((a, b) => a + b, 0) /
                  results.scores.length) *
                100
              ).toFixed(1)}
              %
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
