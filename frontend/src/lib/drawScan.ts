/* Shared canvas rendering for scan annotation (boxes, heatmap, measurements).

One place for how an annotated scan is drawn. Detector, ImageViewer and the
webcam overlay all draw through here; annotation style changes here and nowhere else.
Boxes are xyxy in original image pixels, so the canvas is sized to the image.
*/

import { displayLabel, isTumor, labelNameAt } from "./detections";

export interface ScanOverlays {
  boxes: number[][];
  labels: number[];
  scores: number[];
  label_names?: (string | undefined)[];
  threshold: number;
  heatmap?: HTMLImageElement | null;
  measurements?: { x1: number; y1: number; x2: number; y2: number }[];
  measure_start?: { x1: number; y1: number; x2: number; y2: number } | null;
}

const TUMOR_COLOR = "#10b981";
const NEGATIVE_COLOR = "#64748b";
const MEASURE_COLOR = "#f59e0b";
const MEASURE_LABEL_BG = "rgba(245, 158, 11, 0.9)";
const FONT_FAMILY = "Inter, system-ui, sans-serif";

/** Stroke and font sizes are authored for ~640px images and grow with larger scans. */
function drawScale(canvas: { width: number; height: number }): number {
  return Math.max(1, Math.max(canvas.width, canvas.height) / 640);
}

/** Draw detection boxes with labels onto an already-sized canvas (no background image). */
export function drawDetections(
  ctx: CanvasRenderingContext2D,
  overlays: Pick<ScanOverlays, "boxes" | "labels" | "scores" | "label_names" | "threshold">,
): void {
  const scale = drawScale(ctx.canvas);
  const fontSize = Math.round(14 * scale);
  const barHeight = Math.round(26 * scale);
  const pad = Math.round(6 * scale);
  ctx.lineWidth = Math.max(2, Math.round(3 * scale));
  ctx.font = `bold ${fontSize}px ${FONT_FAMILY}`;

  for (let i = 0; i < overlays.boxes.length; i++) {
    const score = overlays.scores[i] ?? 0;
    if (score < overlays.threshold) continue;
    const [xmin, ymin, xmax, ymax] = overlays.boxes[i];
    const labelName = labelNameAt(overlays, i);
    const color = isTumor(labelName) ? TUMOR_COLOR : NEGATIVE_COLOR;

    ctx.strokeStyle = color;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

    const labelText = `${displayLabel(labelName)}: ${(score * 100).toFixed(1)}%`;
    const textWidth = ctx.measureText(labelText).width;
    // Keep the label bar on the canvas when the box touches the top edge.
    const barTop = ymin - barHeight < 0 ? ymin : ymin - barHeight;
    ctx.fillStyle = color;
    ctx.fillRect(xmin, barTop, textWidth + pad * 2, barHeight);
    ctx.fillStyle = "white";
    ctx.fillText(labelText, xmin + pad, barTop + barHeight / 2 + fontSize * 0.35);
  }
}

export function drawScan(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  overlays: ScanOverlays,
): void {
  // Resizing the canvas resets all context state (font, dash, styles).
  ctx.canvas.width = image.naturalWidth;
  ctx.canvas.height = image.naturalHeight;
  ctx.drawImage(image, 0, 0);

  if (overlays.heatmap) {
    ctx.globalAlpha = 0.5;
    ctx.drawImage(overlays.heatmap, 0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.globalAlpha = 1.0;
  }

  drawDetections(ctx, overlays);

  const scale = drawScale(ctx.canvas);
  ctx.strokeStyle = MEASURE_COLOR;
  ctx.lineWidth = 2 * scale;
  ctx.setLineDash([6 * scale, 4 * scale]);
  ctx.font = `bold ${Math.round(14 * scale)}px ${FONT_FAMILY}`;

  for (const m of overlays.measurements ?? []) {
    ctx.beginPath();
    ctx.moveTo(m.x1, m.y1);
    ctx.lineTo(m.x2, m.y2);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.fillStyle = MEASURE_COLOR;
    ctx.beginPath();
    ctx.arc(m.x1, m.y1, 5 * scale, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(m.x2, m.y2, 5 * scale, 0, Math.PI * 2);
    ctx.fill();

    const dist = Math.sqrt((m.x2 - m.x1) ** 2 + (m.y2 - m.y1) ** 2);
    const mx = (m.x1 + m.x2) / 2;
    const my = (m.y1 + m.y2) / 2;
    const text = `${dist.toFixed(0)}px`;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = MEASURE_LABEL_BG;
    ctx.fillRect(mx - tw / 2 - 4 * scale, my - 12 * scale, tw + 8 * scale, 20 * scale);
    ctx.fillStyle = "white";
    ctx.fillText(text, mx - tw / 2, my + 4 * scale);
    ctx.setLineDash([6 * scale, 4 * scale]);
  }
  ctx.setLineDash([]);

  if (overlays.measure_start) {
    const m = overlays.measure_start;
    ctx.strokeStyle = MEASURE_COLOR;
    ctx.lineWidth = 2 * scale;
    ctx.setLineDash([6 * scale, 4 * scale]);
    ctx.beginPath();
    ctx.moveTo(m.x1, m.y1);
    ctx.lineTo(m.x2, m.y2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
