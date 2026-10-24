import { confirmedDetections, type ImageInfo, type ModelType } from "./api";

export type HistoryModelFilter = "all" | ModelType;
export type HistoryDetectionFilter = "all" | "with" | "without";
export type HistorySort = "newest" | "oldest";

export interface HistoryFilters {
  search: string;
  model: HistoryModelFilter;
  detections: HistoryDetectionFilter;
  sort: HistorySort;
}

export function otherModel(model: string | null | undefined): ModelType {
  return model === "yolov8" ? "faster-rcnn" : "yolov8";
}

export function filterHistory(
  images: ImageInfo[],
  filters: HistoryFilters,
): ImageInfo[] {
  const query = filters.search.trim().toLowerCase();
  const filtered = images.filter((image) => {
    if (query && !image.original_name.toLowerCase().includes(query)) {
      return false;
    }
    if (filters.model !== "all" && image.model_type !== filters.model) {
      return false;
    }
    const hasDetections = confirmedDetections(image).length > 0;
    if (filters.detections === "with" && !hasDetections) return false;
    if (filters.detections === "without" && hasDetections) return false;
    return true;
  });
  return [...filtered].sort((a, b) => {
    const delta = Date.parse(a.uploaded_at) - Date.parse(b.uploaded_at);
    return filters.sort === "oldest" ? delta : -delta;
  });
}
