import { useEffect, useRef, useState } from "react";
import { Download, Eye, Ruler, Thermometer } from "lucide-react";
import type { PredictionResult } from "../lib/api";
import { downloadBlob } from "../lib/download";
import { drawScan } from "../lib/drawScan";
import {
  canvasPointFromEvent,
  clearMeasurements,
  measureClick,
  type MeasureState,
} from "../lib/scanViewer";

type ViewMode = "original" | "annotated" | "split";

const VIEW_MODES: { mode: ViewMode; label: string }[] = [
  { mode: "annotated", label: "Annotated" },
  { mode: "original", label: "Original" },
  { mode: "split", label: "Side-by-side" },
];

const EMPTY_MEASURE: MeasureState = { start: null, list: [] };

interface ScanViewerProps {
  imageUrl: string;
  results: PredictionResult | null;
  threshold: number;
}

export function ScanViewer({ imageUrl, results, threshold }: ScanViewerProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("annotated");
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapImg, setHeatmapImg] = useState<HTMLImageElement | null>(null);
  const [measuring, setMeasuring] = useState(false);
  const [measure, setMeasure] = useState<MeasureState>(EMPTY_MEASURE);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    setViewMode("annotated");
    setShowHeatmap(false);
    setMeasuring(false);
    setMeasure(EMPTY_MEASURE);
  }, [imageUrl]);

  // Decode the heatmap once per result instead of on every redraw.
  const heatmap = results?.heatmap;
  useEffect(() => {
    if (!heatmap) {
      setHeatmapImg(null);
      return;
    }
    let cancelled = false;
    const img = new window.Image();
    img.onload = () => {
      if (!cancelled) setHeatmapImg(img);
    };
    img.src = `data:image/png;base64,${heatmap}`;
    return () => {
      cancelled = true;
    };
  }, [heatmap]);

  const render = (current?: { x: number; y: number }) => {
    const img = imgRef.current;
    const ctx = canvasRef.current?.getContext("2d");
    if (!img || !img.naturalWidth || !ctx) return;
    const start = measure.start;
    drawScan(ctx, img, {
      boxes: results?.boxes ?? [],
      labels: results?.labels ?? [],
      scores: results?.scores ?? [],
      label_names: results?.label_names,
      threshold,
      heatmap: showHeatmap ? heatmapImg : null,
      measurements: measure.list,
      measure_start: start
        ? { x1: start.x, y1: start.y, x2: current?.x ?? start.x, y2: current?.y ?? start.y }
        : null,
    });
  };

  useEffect(() => {
    render();
    // render reads the same state listed here; it is recreated every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results, threshold, viewMode, measure, showHeatmap, heatmapImg]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!measuring || !canvasRef.current) return;
    setMeasure((prev) => measureClick(prev, canvasPointFromEvent(e, canvasRef.current!)));
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!measuring || !measure.start || !canvasRef.current) return;
    render(canvasPointFromEvent(e, canvasRef.current));
  };

  const downloadAnnotated = () => {
    canvasRef.current?.toBlob((blob) => {
      if (blob) downloadBlob(blob, "annotated-scan.png");
    });
  };

  const showCanvas = Boolean(results) && viewMode !== "original";
  const showImage = !results || viewMode !== "annotated";
  const half = viewMode === "split" ? { maxWidth: "50%" } : undefined;

  return (
    <div className="text-center">
      <div className="preview-container" style={viewMode === "split" ? { gap: "0.5rem" } : undefined}>
        <img
          ref={imgRef}
          src={imageUrl}
          alt="Preview"
          style={{ display: showImage ? "block" : "none", maxWidth: "100%", ...half }}
          onLoad={() => render()}
        />
        <canvas
          ref={canvasRef}
          style={{
            display: showCanvas ? "block" : "none",
            cursor: measuring ? "crosshair" : "default",
            ...half,
          }}
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMove}
        />
      </div>

      {results && (
        <div className="view-toolbar">
          <div className="toolbar-group">
            <Eye size={14} />
            {VIEW_MODES.map(({ mode, label }) => (
              <button
                key={mode}
                className={`btn btn-tiny ${viewMode === mode ? "active" : ""}`}
                onClick={() => setViewMode(mode)}
              >
                {label}
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
              setMeasure((prev) => ({ ...prev, start: null }));
            }}
          >
            <Ruler size={14} /> {measuring ? "Done Measuring" : "Measure"}
          </button>

          {measure.list.length > 0 && (
            <button className="btn btn-tiny" onClick={() => setMeasure(clearMeasurements())}>
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
