import React, { useEffect, useRef, useState } from "react";
import { X, Download, FileText, AlertCircle } from "lucide-react";
import {
  authFetch,
  getReportUrl,
  listImages,
  detectionsToPredictionResult,
  type PredictionResult,
} from "../lib/api";
import { drawScan } from "../lib/drawScan";

interface ImageViewerProps {
  imageId: number;
  results: PredictionResult | null;
  onClose: () => void;
}

export function ImageViewer({ imageId, results, onClose }: ImageViewerProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [threshold, setThreshold] = useState(0.5);
  const [fallbackResults, setFallbackResults] = useState<PredictionResult | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // When opened via URL (deep link / refresh) no in-memory results exist;
  // fall back to the detections persisted on the backend.
  useEffect(() => {
    if (results) return;
    let cancelled = false;
    (async () => {
      try {
        const images = await listImages();
        const img = images.find((i) => i.id === imageId);
        if (!cancelled && img) {
          setFallbackResults(detectionsToPredictionResult(img.detections));
        }
      } catch {
        // Annotations are optional; the raw image still renders.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [imageId, results]);

  const effectiveResults = results ?? fallbackResults;
  const visibleBoxes =
    effectiveResults?.boxes.filter((_, i) => (effectiveResults.scores[i] ?? 0) >= threshold) ??
    [];
  const visibleLabels =
    effectiveResults?.label_names?.filter(
      (_, i) => (effectiveResults.scores[i] ?? 0) >= threshold,
    ) ?? [];

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
    if (!effectiveResults || !canvasRef.current || !imgRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    const img = imgRef.current;
    if (!ctx) return;

    const onLoad = () => {
      drawScan(ctx, img, {
        boxes: effectiveResults.boxes,
        labels: effectiveResults.labels,
        scores: effectiveResults.scores,
        label_names: effectiveResults.label_names,
        threshold,
      });
    };

    if (img.complete && img.naturalWidth) {
      onLoad();
    } else {
      img.onload = onLoad;
    }
  }, [effectiveResults, blobUrl, threshold]);

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
          <label className="threshold-label" style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: "0.5rem" }}>
            Conf: <strong>{(threshold * 100).toFixed(0)}%</strong>
            <input type="range" min="0.05" max="0.95" step="0.05" value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="threshold-slider" style={{ width: "7rem" }} />
          </label>
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
        {visibleBoxes.length > 0 && (
          <div className="viewer-summary">
            <p><strong>{visibleBoxes.length}</strong> detection{visibleBoxes.length !== 1 ? "s" : ""}</p>
            {visibleLabels.length > 0 && (
              <p>Top class: <strong>{visibleLabels[0]}</strong></p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
