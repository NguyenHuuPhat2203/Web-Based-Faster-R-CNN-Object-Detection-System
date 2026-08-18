import { useEffect, useRef, useState } from "react";
import { Download, Eye, Ruler, Thermometer } from "lucide-react";
import type { PredictionResult } from "../lib/api";
import { drawScan } from "../lib/drawScan";
import {
  canvasPointFromEvent,
  clearMeasurements,
  measureClick,
  type MeasureState,
} from "../lib/scanViewer";

type ViewMode = "original" | "annotated" | "split";

interface ScanViewerProps {
  imageUrl: string;
  results: PredictionResult | null;
  threshold: number;
}

export function ScanViewer({ imageUrl, results, threshold }: ScanViewerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("annotated");
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [measuring, setMeasuring] = useState(false);
  const [measure, setMeasure] = useState<MeasureState>({ start: null, list: [] });
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    setViewMode("annotated");
    setShowHeatmap(false);
    setMeasuring(false);
    setMeasure({ start: null, list: [] });
  }, [imageUrl]);

  const overlays = (current?: { x: number; y: number }) => ({
    boxes: results?.boxes ?? [],
    labels: results?.labels ?? [],
    scores: results?.scores ?? [],
    label_names: results?.label_names,
    threshold,
    measurements: measure.list,
    measure_start: current
      ? {
          x1: measure.start?.x ?? current.x,
          y1: measure.start?.y ?? current.y,
          x2: current.x,
          y2: current.y,
        }
      : measure.start
        ? { x1: measure.start.x, y1: measure.start.y, x2: measure.start.x, y2: measure.start.y }
        : null,
  });

  const render = (current?: { x: number; y: number }) => {
    if (!canvasRef.current || !imgRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    if (viewMode === "original") {
      drawScan(ctx, imgRef.current, { boxes: [], labels: [], scores: [], threshold });
      return;
    }
    drawScan(ctx, imgRef.current, overlays(current));
  };

  useEffect(() => {
    render();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results, threshold, viewMode, measure, showHeatmap]);

  useEffect(() => {
    if (!showHeatmap || !results?.heatmap || !canvasRef.current || !imgRef.current) return;
    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;
    const hmImg = new window.Image();
    hmImg.onload = () => {
      if (viewMode === "original") {
        drawScan(ctx, imgRef.current!, { boxes: [], labels: [], scores: [], threshold });
        return;
      }
      drawScan(ctx, imgRef.current!, { ...overlays(), heatmap: hmImg });
    };
    hmImg.src = `data:image/png;base64,${results.heatmap}`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showHeatmap, results, viewMode, measure, threshold]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!measuring || !canvasRef.current) return;
    setMeasure((prev) => measureClick(prev, canvasPointFromEvent(e, canvasRef.current!)));
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!measuring || !measure.start || !canvasRef.current) return;
    render(canvasPointFromEvent(e, canvasRef.current));
  };

  const downloadAnnotated = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `annotated-scan.png`;
    canvas.toBlob((blob) => {
      if (blob) {
        link.href = URL.createObjectURL(blob);
        link.click();
      }
    });
  };

  return (
    <div className="text-center">
      <div className="preview-container">
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Preview"
          style={{
            display: results && viewMode !== "original" ? "none" : "block",
            maxWidth: "100%",
          }}
          onLoad={() => render()}
        />
        <canvas
          ref={canvasRef}
          style={{
            display: results && viewMode !== "original" ? "block" : "none",
            cursor: measuring ? "crosshair" : "default",
          }}
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMove}
        />
      </div>

      {results && (
        <div className="view-toolbar">
          <div className="toolbar-group">
            <Eye size={14} />
            {(["annotated", "original", "split"] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                className={`btn btn-tiny ${viewMode === mode ? "active" : ""}`}
                onClick={() => setViewMode(mode)}
              >
                {mode === "annotated" ? "Annotated" : mode === "original" ? "Original" : "Side-by-side"}
              </button>
            ))}
          </div>

          {results.heatmap && (
            <button
              className={`btn btn-tiny ${showHeatmap ? "active" : ""}`}
              onClick={() => setShowHeatmap(!showHeatmap)}
            >
              <Thermometer size={14} /> {showHeatmap ? "Hide Heatmap" : "Heatmap"}
            </button>
          )}

          <button
            className={`btn btn-tiny ${measuring ? "active" : ""}`}
            onClick={() => {
              setMeasuring(!measuring);
              if (!measuring) setMeasure((prev) => ({ ...prev, start: null }));
            }}
          >
            <Ruler size={14} /> {measuring ? "Done Measuring" : "Measure"}
          </button>

          {measure.list.length > 0 && (
            <button
              className="btn btn-tiny"
              onClick={() => setMeasure((prev) => clearMeasurements(prev))}
            >
              Clear Lines
            </button>
          )}

          <button className="btn btn-tiny" onClick={downloadAnnotated}>
            <Download size={14} /> Download
          </button>
        </div>
      )}
    </div>
  );
}
