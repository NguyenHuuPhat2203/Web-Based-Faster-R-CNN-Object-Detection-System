import { describe, expect, it } from "vitest";
import {
  canvasPointFromEvent,
  clearMeasurements,
  measureClick,
  type MeasureState,
} from "./scanViewer";

const empty: MeasureState = { start: null, list: [] };

const canvas = {
  width: 200,
  height: 100,
  getBoundingClientRect: () => ({ left: 10, top: 20, width: 100, height: 50 }),
};

describe("canvasPointFromEvent", () => {
  it("maps client coordinates to image pixels with scale", () => {
    const p = canvasPointFromEvent({ clientX: 60, clientY: 45 }, canvas);
    expect(p).toEqual({ x: 100, y: 50 });
  });
});

describe("measureClick", () => {
  it("stores the first click as the start point", () => {
    const next = measureClick(empty, { x: 10, y: 10 });
    expect(next.start).toEqual({ x: 10, y: 10 });
    expect(next.list).toEqual([]);
  });

  it("appends a measurement and clears the start on the second click", () => {
    const afterFirst = measureClick(empty, { x: 0, y: 0 });
    const afterSecond = measureClick(afterFirst, { x: 40, y: 30 });
    expect(afterSecond.start).toBeNull();
    expect(afterSecond.list).toEqual([{ x1: 0, y1: 0, x2: 40, y2: 30 }]);
  });

  it("chains multiple measurements", () => {
    const a = measureClick(empty, { x: 1, y: 1 });
    const b = measureClick(a, { x: 5, y: 5 });
    const c = measureClick(b, { x: 9, y: 9 });
    const d = measureClick(c, { x: 13, y: 13 });
    expect(d.list).toHaveLength(2);
  });
});

describe("clearMeasurements", () => {
  it("resets the list and the start point", () => {
    const withStart = measureClick(empty, { x: 5, y: 5 });
    expect(clearMeasurements(withStart)).toEqual(empty);
  });
});
