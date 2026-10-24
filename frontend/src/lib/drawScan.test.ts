import { describe, expect, it } from "vitest";
import { drawScan, type ScanOverlays } from "./drawScan";

function fakeCtx() {
  const calls: string[] = [];
  const ctx = {
    canvas: { width: 0, height: 0 },
    strokeStyle: "",
    fillStyle: "",
    lineWidth: 0,
    font: "",
    globalAlpha: 1,
    drawImage: () => {
      calls.push("drawImage");
    },
    strokeRect: (x: number, y: number, w: number, h: number) => {
      calls.push(`strokeRect:${x}:${y}:${w}:${h}`);
    },
    fillRect: (x: number, y: number, w: number, h: number) => {
      calls.push(`fillRect:${x}:${y}:${w}:${h}`);
    },
    fillText: (t: string, x: number, y: number) => {
      calls.push(`fillText:${t}:${x}:${y}`);
    },
    measureText: (t: string) => ({ width: t.length * 8 }),
    setLineDash: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    stroke: () => {
      calls.push("stroke");
    },
    arc: () => {},
    fill: () => {},
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls };
}

const img = { naturalWidth: 100, naturalHeight: 50 } as HTMLImageElement;

function overlays(over: Partial<ScanOverlays> = {}): ScanOverlays {
  return {
    boxes: [
      [10, 20, 60, 80],
      [5, 5, 15, 15],
    ],
    labels: [1, 1],
    scores: [0.9, 0.3],
    threshold: 0.5,
    ...over,
  };
}

describe("drawScan", () => {
  it("sizes the canvas to the image and draws the image", () => {
    const { ctx, calls } = fakeCtx();
    drawScan(ctx, img, overlays());
    expect(ctx.canvas.width).toBe(100);
    expect(ctx.canvas.height).toBe(50);
    expect(calls[0]).toBe("drawImage");
  });

  it("skips detections below the threshold", () => {
    const { ctx, calls } = fakeCtx();
    drawScan(ctx, img, overlays());
    // box 1 (0.9) drawn, box 2 (0.3) filtered: exactly one strokeRect
    expect(calls.filter((c) => c.startsWith("strokeRect")).length).toBe(1);
  });

  it("draws the label bar above the box when there is room", () => {
    const { ctx, calls } = fakeCtx();
    drawScan(ctx, img, overlays({ boxes: [[10, 40, 60, 80]], scores: [0.9] }));
    // bar: fillRect(xmin, ymin - barHeight, textWidth + 2 * pad, barHeight)
    const bar = calls.find((c) => c.startsWith("fillRect:"));
    expect(bar).toBe("fillRect:10:14:124:26"); // textWidth=112 (14 chars x 8px) + 12
  });

  it("keeps the label bar on the canvas when the box touches the top edge", () => {
    const { ctx, calls } = fakeCtx();
    drawScan(ctx, img, overlays({ boxes: [[10, 5, 60, 40]], scores: [0.9] }));
    const bar = calls.find((c) => c.startsWith("fillRect:"));
    expect(bar).toBe("fillRect:10:5:124:26");
  });

  it("draws dashed measurement lines with px labels", () => {
    const { ctx, calls } = fakeCtx();
    drawScan(ctx, img, overlays({ measurements: [{ x1: 0, y1: 0, x2: 40, y2: 30 }] }));
    const strokes = calls.filter((c) => c === "stroke");
    expect(strokes.length).toBe(1);
    expect(calls.some((c) => c.startsWith("fillText:50px"))).toBe(true);
  });

  it("draws the live measurement preview line", () => {
    const { calls } = fakeCtx();
    const { ctx, calls: previewCalls } = fakeCtx();
    drawScan(
      ctx,
      img,
      overlays({ measure_start: { x1: 5, y1: 5, x2: 20, y2: 20 } }),
    );
    calls.splice(0, 0, ...previewCalls);
    expect(calls.filter((c) => c === "stroke").length).toBe(1);
  });

  it("overlays the heatmap at half opacity before annotations", () => {
    const { ctx, calls } = fakeCtx();
    const heatmap = { naturalWidth: 1, naturalHeight: 1 } as HTMLImageElement;
    drawScan(ctx, img, overlays({ heatmap }));
    expect(ctx.globalAlpha).toBe(1);
    expect(calls.filter((c) => c === "drawImage").length).toBe(2);
  });
});
