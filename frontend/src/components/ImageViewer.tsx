import React, { useEffect, useRef, useState } from "react";
import { X, Download, FileText, AlertCircle } from "lucide-react";
import {
  authFetch,
  getReportUrl,
  type PredictionResult,
} from "../lib/api";

interface ImageViewerProps {
  imageId: number;
  results: PredictionResult | null;
  onClose: () => void;
}

export function ImageViewer({ imageId, results, onClose }: ImageViewerProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await authFetch(`/images/${imageId}`);
        if (!res.ok) throw new Error("Failed to load image");
        const blob = await res.blob();
        if (cancelled) return;
        setBlobUrl(URL.createObjectURL(blob));
      } catch (err: any) {
        if (!cancelled) setError(err.message || "Failed to load image");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [imageId]);

  // Draw annotations once the image loads
  useEffect(() => {
    if (!results || !canvasRef.current || !imgRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;
    if (!ctx) return;

    const onLoad = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      if (!results.boxes.length) return;

      ctx.lineWidth = 4;
      ctx.font = "bold 22px Inter, system-ui, sans-serif";

      results.boxes.forEach((box, i) => {
        const [xmin, ymin, xmax, ymax] = box;
        const score = results.scores[i];
        const label = results.label_names?.[i] || `Class ${results.labels[i]}`;

        ctx.strokeStyle = "#10b981";
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        const labelText = `${label}: ${(score * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = "#10b981";
        ctx.fillRect(xmin, ymin - 34, textWidth + 14, 34);
        ctx.fillStyle = "white";
        ctx.fillText(labelText, xmin + 7, ymin - 9);
      });
    };

    if (img.complete && img.naturalWidth) {
      onLoad();
    } else {
      img.onload = onLoad;
    }
  }, [results, blobUrl]);

  const handleDownloadReport = async () => {
    const url = getReportUrl(imageId);
    const resp = await authFetch(url);
    if (!resp.ok) { setError("Failed to generate report"); return; }
    const blob = await resp.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `report-${imageId}.pdf`;
    link.click();
  };

  const handleDownloadAnnotated = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.toBlob((blob) => {
      if (blob) {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `scan-${imageId}.png`;
        link.click();
      }
    });
  };

  return (
    <div className="viewer-overlay" onClick={onClose}>
      <div className="viewer-modal" onClick={(e) => e.stopPropagation()}>
        {/* Close button at top-left */}
        <button className="viewer-close" onClick={onClose} title="Close">
          <X size={20} />
        </button>

        {/* Toolbar */}
        <div className="viewer-toolbar">
          <button className="btn btn-tiny" onClick={handleDownloadAnnotated} title="Download Annotated">
            <Download size={14} /> Download
          </button>
          <button className="btn btn-tiny" onClick={handleDownloadReport} title="Download Report">
            <FileText size={14} /> Report
          </button>
        </div>

        {/* Image area */}
        <div className="viewer-content">
          {loading && (
            <div className="viewer-status">
              <div className="spinner" />
              <p>Loading image&hellip;</p>
            </div>
          )}
          {error && (
            <div className="viewer-status">
              <AlertCircle size={24} />
              <p>{error}</p>
            </div>
          )}
          {blobUrl && (
            <>
              <img
                ref={imgRef}
                src={blobUrl}
                alt="Scan"
                style={{ display: "none" }}
              />
              <canvas ref={canvasRef} className="viewer-canvas" />
            </>
          )}
        </div>

        {/* Detection summary */}
        {results && results.boxes.length > 0 && (
          <div className="viewer-summary">
            <p><strong>{results.boxes.length}</strong> detection{results.boxes.length !== 1 ? "s" : ""}</p>
            {results.label_names?.length > 0 && (
              <p>Top class: <strong>{results.label_names[0]}</strong></p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
