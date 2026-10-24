/* Client-side view of a PredictionResult: confidence filtering and label semantics.

The server returns every detection with score >= min(requested threshold, 0.1),
so the confidence slider can filter locally anywhere in [CONFIDENCE_MIN, CONFIDENCE_MAX].
Labels follow the dataset vocabulary ("negative" / "positive"); a box is not
necessarily a tumor.
*/

import type { PredictionResult } from "./api";

/** Lowest score the server is guaranteed to return; the slider must not go below it. */
export const CONFIDENCE_MIN = 0.1;
export const CONFIDENCE_MAX = 0.95;
export const CONFIDENCE_STEP = 0.05;
export const DEFAULT_CONFIDENCE = 0.5;

export interface VisibleDetection {
  index: number;
  labelName: string;
  score: number;
}

export function labelNameAt(
  result: { labels: number[]; label_names?: (string | undefined)[] },
  index: number,
): string {
  return result.label_names?.[index] || `Class ${result.labels[index] ?? 0}`;
}

/** Findings that are not explicitly negative count as tumors (older records say "Brain tumor"). */
export function isTumor(labelName: string): boolean {
  return labelName.toLowerCase() !== "negative";
}

export function displayLabel(labelName: string): string {
  const lower = labelName.toLowerCase();
  if (lower === "negative") return "Negative";
  if (lower === "positive") return "Tumor";
  return labelName;
}

export function visibleDetections(
  result: PredictionResult,
  threshold: number,
): VisibleDetection[] {
  const visible: VisibleDetection[] = [];
  result.boxes.forEach((_, index) => {
    const score = result.scores[index] ?? 0;
    if (score >= threshold) {
      visible.push({ index, labelName: labelNameAt(result, index), score });
    }
  });
  return visible;
}
