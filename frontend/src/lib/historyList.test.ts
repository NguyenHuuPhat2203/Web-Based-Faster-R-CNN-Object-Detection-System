import { describe, expect, it } from "vitest";
import { confirmedDetections, type ImageInfo } from "./api";
import { filterHistory, otherModel } from "./historyList";

function img(over: Partial<ImageInfo> & Pick<ImageInfo, "id" | "original_name">): ImageInfo {
  return {
    mime_type: "image/png",
    uploaded_at: "2026-01-02T00:00:00Z",
    width: 8,
    height: 8,
    file_size: 10,
    model_type: "faster-rcnn",
    model_version: "v1",
    threshold: 0.5,
    detections: [],
    ...over,
  };
}

const rows: ImageInfo[] = [
  img({
    id: 1,
    original_name: "t1_scan.png",
    uploaded_at: "2026-01-01T00:00:00Z",
    model_type: "faster-rcnn",
    detections: [{ label: 1, label_name: "t", score: 0.9, x1: 0, y1: 0, x2: 1, y2: 1 }],
  }),
  img({
    id: 2,
    original_name: "flair.png",
    uploaded_at: "2026-01-03T00:00:00Z",
    model_type: "yolov8",
    detections: [],
  }),
];

describe("filterHistory", () => {
  it("filters by filename", () => {
    const out = filterHistory(rows, {
      search: "flair",
      model: "all",
      detections: "all",
      sort: "newest",
    });
    expect(out.map((r) => r.id)).toEqual([2]);
  });

  it("filters by model and detection presence", () => {
    const out = filterHistory(rows, {
      search: "",
      model: "faster-rcnn",
      detections: "with",
      sort: "newest",
    });
    expect(out.map((r) => r.id)).toEqual([1]);
  });

  it("sorts oldest first", () => {
    const out = filterHistory(rows, {
      search: "",
      model: "all",
      detections: "all",
      sort: "oldest",
    });
    expect(out.map((r) => r.id)).toEqual([1, 2]);
  });
});

describe("otherModel", () => {
  it("swaps Faster R-CNN and YOLO", () => {
    expect(otherModel("faster-rcnn")).toBe("yolov8");
    expect(otherModel("yolov8")).toBe("faster-rcnn");
    expect(otherModel(null)).toBe("yolov8");
  });
});

describe("confirmedDetections", () => {
  const detection = (score: number) => ({
    label: 2, label_name: "positive", score, x1: 0, y1: 0, x2: 1, y2: 1,
  });
  const image = (threshold: number | null) =>
    ({ threshold, detections: [detection(0.9), detection(0.2)] }) as unknown as ImageInfo;

  it("drops detections stored below the image threshold", () => {
    expect(confirmedDetections(image(0.5))).toHaveLength(1);
  });

  it("keeps everything when the threshold is unknown", () => {
    expect(confirmedDetections(image(null))).toHaveLength(2);
  });
});
