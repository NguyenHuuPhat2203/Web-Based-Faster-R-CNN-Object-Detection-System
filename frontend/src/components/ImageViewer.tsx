import { useEffect, useRef, useState } from "react";
import { X, Download, FileText, AlertCircle } from "lucide-react";
import {
  messageOf,
  downloadReport,
  fetchImageBlob,
  listImages,
  imageToPredictionResult,
  type PredictionResult,
} from "../lib/api";
import { DEFAULT_CONFIDENCE, displayLabel, visibleDetections } from "../lib/detections";
import { downloadBlob } from "../lib/download";
import { drawScan } from "../lib/drawScan";
import { ConfidenceSlider } from "./ConfidenceSlider";

interface ImageViewerProps {
  imageId: number;
  results: PredictionResult | null;
  onClose: () => void;
}

export function ImageViewer({ imageId, results, onClose }: ImageViewerProps) {
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [imageReady, setImageReady] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [threshold, setThreshold] = useState(DEFAULT_CONFIDENCE);
  const [fallbackResults, setFallbackResults] = useState<PredictionResult | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  // When opened via URL (deep link / refresh) no in-memory results exist;
  // fall back to the detections persisted on the backend.
  useEffect(() => {
    setFallbackResults(null);
    if (results) return;
    let cancelled = false;
    listImages()
      .then((images) => {
        const img = images.find((i) => i.id === imageId);
        if (!cancelled && img) {
          setFallbackResults(imageToPredictionResult(img));
        }
      })
      .catch(() => {
        // Annotations are optional; the raw image still renders.
      });
    return () => {
      cancelled = true;
    };
  }, [imageId, results]);

  const effectiveResults = results ?? fallbackResults;
  const visible = effectiveResults ? visibleDetections(effectiveResults, threshold) : [];
  const topDetection = visible.reduce<(typeof visible)[number] | null>(
    (best, d) => (!best || d.score > best.score ? d : best),
    null,
  );

  // Protected image: fetch with the Bearer token and own the object URL's lifetime.
  useEffect(() => {
    let cancelled = false;
    let url: string | null = null;
    setBlobUrl(null);
    setImageReady(false);
    setError("");
    setLoading(true);
    fetchImageBlob(imageId)
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setBlobUrl(url);
      })
      .catch((err) => {
        if (!cancelled) setError(messageOf(err, "Failed to load image"));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [imageId]);

  // Draw the scan (annotated when results exist) once the image has decoded.
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    const ctx = canvas?.getContext("2d");
    if (!imageReady || !img || !ctx) return;
    drawScan(ctx, img, {
      boxes: effectiveResults?.boxes ?? [],
      labels: effectiveResults?.labels ?? [],
      scores: effectiveResults?.scores ?? [],
      label_names: effectiveResults?.label_names,
      threshold,
    });
  }, [effectiveResults, imageReady, threshold]);

  const handleDownloadReport = async () => {
    try {
      await downloadReport(imageId);
    } catch (err) {
      setError(messageOf(err, "Failed to generate report"));
    }
  };

  const handleDownloadAnnotated = () => {
    canvasRef.current?.toBlob((blob) => {
      if (blob) downloadBlob(blob, `scan-${imageId}.png`);
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
            <ConfidenceSlider value={threshold} onChange={setThreshold} style={{ width: "7rem" }} />
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
                onLoad={() => setImageReady(true)}
              />
              <canvas ref={canvasRef} className="viewer-canvas" />
            </>
          )}
        </div>

        {/* Detection summary */}
        {topDetection && (
          <div className="viewer-summary">
            <p><strong>{visible.length}</strong> detection{visible.length !== 1 ? "s" : ""}</p>
            <p>Top class: <strong>{displayLabel(topDetection.labelName)}</strong></p>
          </div>
        )}
      </div>
    </div>
  );
}
