/* Shared canvas rendering for scan annotation (boxes, heatmap, measurements).

One place for how an annotated scan is drawn. Detector and ImageViewer both
call drawScan; annotation style changes here and nowhere else.
*/

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

const BOX_COLOR = "#10b981";
const MEASURE_COLOR = "#f59e0b";
const MEASURE_LABEL_BG = "rgba(245, 158, 11, 0.9)";

export function drawScan(
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  overlays: ScanOverlays,
): void {
  ctx.canvas.width = image.naturalWidth;
  ctx.canvas.height = image.naturalHeight;
  ctx.drawImage(image, 0, 0);

  if (overlays.heatmap) {
    ctx.globalAlpha = 0.5;
    ctx.drawImage(overlays.heatmap, 0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.globalAlpha = 1.0;
  }

  for (let i = 0; i < overlays.boxes.length; i++) {
    const score = overlays.scores[i] ?? 0;
    if (score < overlays.threshold) continue;
    const [xmin, ymin, xmax, ymax] = overlays.boxes[i];
    const label = overlays.label_names?.[i] || `Class ${overlays.labels[i] ?? 0}`;

    ctx.lineWidth = 4;
    ctx.strokeStyle = BOX_COLOR;
    ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

    const labelText = `${label}: ${(score * 100).toFixed(1)}%`;
    const textWidth = ctx.measureText(labelText).width;
    ctx.fillStyle = BOX_COLOR;
    ctx.fillRect(xmin, ymin - 34, textWidth + 14, 34);
    ctx.fillStyle = "white";
    ctx.fillText(labelText, xmin + 7, ymin - 9);
  }

  ctx.strokeStyle = MEASURE_COLOR;
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.font = "bold 14px Inter, system-ui, sans-serif";

  for (const m of overlays.measurements ?? []) {
    ctx.beginPath();
    ctx.moveTo(m.x1, m.y1);
    ctx.lineTo(m.x2, m.y2);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.fillStyle = MEASURE_COLOR;
    ctx.beginPath();
    ctx.arc(m.x1, m.y1, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(m.x2, m.y2, 5, 0, Math.PI * 2);
    ctx.fill();

    const dist = Math.sqrt((m.x2 - m.x1) ** 2 + (m.y2 - m.y1) ** 2);
    const mx = (m.x1 + m.x2) / 2;
    const my = (m.y1 + m.y2) / 2;
    const text = `${dist.toFixed(0)}px`;
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = MEASURE_LABEL_BG;
    ctx.fillRect(mx - tw / 2 - 4, my - 12, tw + 8, 20);
    ctx.fillStyle = "white";
    ctx.fillText(text, mx - tw / 2, my + 4);
    ctx.setLineDash([6, 4]);
  }
  ctx.setLineDash([]);

  if (overlays.measure_start) {
    const m = overlays.measure_start;
    ctx.strokeStyle = MEASURE_COLOR;
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(m.x1, m.y1);
    ctx.lineTo(m.x2, m.y2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
