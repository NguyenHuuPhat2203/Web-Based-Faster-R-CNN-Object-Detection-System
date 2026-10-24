import { describe, expect, it } from "vitest";
import {
  displayLabel,
  isTumor,
  visibleDetections,
} from "./detections";

const result = {
  boxes: [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]],
  labels: [2, 1, 2],
  scores: [0.9, 0.6, 0.2],
  label_names: ["positive", "negative", "positive"],
};

describe("visibleDetections", () => {
  it("keeps detections at or above the threshold, preserving indices", () => {
    const visible = visibleDetections(result, 0.6);
    expect(visible.map((d) => d.index)).toEqual([0, 1]);
    expect(visible[1].labelName).toBe("negative");
  });

  it("falls back to the numeric class when names are missing", () => {
    const visible = visibleDetections({ ...result, label_names: [] }, 0.5);
    expect(visible[0].labelName).toBe("Class 2");
  });
});

describe("label semantics", () => {
  it("treats negative as non-tumor and everything else as tumor", () => {
    expect(isTumor("negative")).toBe(false);
    expect(isTumor("positive")).toBe(true);
    expect(isTumor("Brain tumor")).toBe(true);
  });

  it("maps dataset labels to display text", () => {
    expect(displayLabel("positive")).toBe("Tumor");
    expect(displayLabel("negative")).toBe("Negative");
    expect(displayLabel("Brain tumor")).toBe("Brain tumor");
  });
});
