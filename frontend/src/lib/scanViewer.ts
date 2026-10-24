/* Pure logic for the scan viewer: measurement state transitions.

Coordinate mapping and measurement click handling live here so they are
testable without a DOM; the ScanViewer component stays thin.
*/

export interface Measurement {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface MeasureState {
  start: { x: number; y: number } | null;
  list: Measurement[];
}

interface PointLike {
  clientX: number;
  clientY: number;
}

interface CanvasLike {
  width: number;
  height: number;
  getBoundingClientRect(): { left: number; top: number; width: number; height: number };
}

export function canvasPointFromEvent(
  e: PointLike,
  canvas: CanvasLike,
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

export function measureClick(
  state: MeasureState,
  point: { x: number; y: number },
): MeasureState {
  if (state.start === null) {
    return { ...state, start: point };
  }
  return {
    start: null,
    list: [
      ...state.list,
      { x1: state.start.x, y1: state.start.y, x2: point.x, y2: point.y },
    ],
  };
}

export function clearMeasurements(): MeasureState {
  return { start: null, list: [] };
}
