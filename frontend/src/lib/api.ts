import { API_BASE, authFetch, processOAuthTokens, setTokens } from "./session";

export { processOAuthTokens };

/* ---------- Auth API calls ---------- */

export interface UserInfo {
  id: number;
  email: string;
  username: string;
  created_at: string;
}

export interface LoginResponse {
  id: number;
  email: string;
  username: string;
  created_at: string;
  access_token: string;
  refresh_token: string;
}

export async function register(
  email: string,
  username: string,
  password: string,
): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, username, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Registration failed" }));
    throw new Error(err.detail || "Registration failed");
  }
  const data: LoginResponse = await res.json();
  setTokens(data.access_token, data.refresh_token);
  return data;
}

export async function login(
  email: string,
  password: string,
): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Login failed" }));
    throw new Error(err.detail || "Login failed");
  }
  const data: LoginResponse = await res.json();
  setTokens(data.access_token, data.refresh_token);
  return data;
}

export function getGoogleAuthUrl(): string {
  return `${API_BASE}/auth/google`;
}

/* ---------- Predict API ---------- */

export interface PredictionResult {
  boxes: number[][];
  labels: number[];
  scores: number[];
  label_names: string[];
  heatmap?: string | null; // base64 PNG, only for Faster R-CNN
  model_type?: string | null;
  model_version?: string | null;
}

export type ModelType = "faster-rcnn" | "yolov8";

export type ApiVersion = "v1" | "v2";

/** Backend API contract version. Default v2; set VITE_API_VERSION=v1 to rollback. */
export const API_VERSION: ApiVersion =
  import.meta.env.VITE_API_VERSION === "v1" ? "v1" : "v2";

export interface PredictOptions {
  /** Grad-CAM for Faster R-CNN on v2 only. Webcam should pass false. */
  explain?: boolean;
}

function formatPredictError(message: string, res: Response): Error {
  const correlationId = res.headers.get("x-correlation-id");
  if (correlationId) {
    return new Error(`${message} (correlation_id=${correlationId})`);
  }
  return new Error(message);
}

function unwrapV2Prediction(data: unknown, res: Response): PredictionResult {
  if (!data || typeof data !== "object") {
    throw formatPredictError("Invalid prediction response", res);
  }
  const body = data as {
    status?: string;
    result?: PredictionResult;
    error?: { code?: string; message?: string };
  };
  if (body.status === "ok" && body.result) {
    return body.result;
  }
  if (body.status === "error") {
    const code = body.error?.code ? `[${body.error.code}] ` : "";
    const message = body.error?.message || "Prediction failed";
    throw formatPredictError(`${code}${message}`, res);
  }
  throw formatPredictError("Unexpected prediction response shape", res);
}

export async function predict(
  file: File,
  threshold = 0.5,
  model: ModelType = "faster-rcnn",
  options: PredictOptions = {},
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);

  const explain = options.explain ?? false;
  let endpoint: string;

  if (API_VERSION === "v2") {
    const params = new URLSearchParams({
      threshold: String(threshold),
    });
    if (model === "faster-rcnn") {
      params.set("explain", explain ? "true" : "false");
      endpoint = `/v2/predict?${params.toString()}`;
    } else {
      endpoint = `/v2/predict/yolo?${params.toString()}`;
    }
  } else {
    // v1 bare contract — threshold in form body (legacy)
    formData.append("threshold", String(threshold));
    endpoint = model === "yolov8" ? "/predict-yolo" : "/predict";
  }

  const res = await authFetch(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Prediction failed" }));
    const detail =
      typeof err.detail === "string"
        ? err.detail
        : Array.isArray(err.detail)
          ? err.detail.map((d: { msg?: string }) => d.msg).filter(Boolean).join("; ")
          : "Prediction failed";
    throw formatPredictError(detail || "Prediction failed", res);
  }

  const data = await res.json();
  if (API_VERSION === "v2") {
    return unwrapV2Prediction(data, res);
  }
  return data as PredictionResult;
}

/* ---------- Images API ---------- */

export interface ImageInfo {
  id: number;
  original_name: string;
  mime_type: string;
  uploaded_at: string;
  width: number | null;
  height: number | null;
  file_size: number | null;
  model_type: string | null;
  model_version: string | null;
  threshold: number | null;
  detections: DetectionInfo[];
}

export interface DetectionInfo {
  label: number;
  label_name: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Convert the backend's detections array to the PredictionResult format used by the Detector. */
export function detectionsToPredictionResult(detections: DetectionInfo[]): PredictionResult {
  return {
    boxes: detections.map((d) => [d.x1, d.y1, d.x2, d.y2]),
    labels: detections.map((d) => d.label),
    scores: detections.map((d) => d.score),
    label_names: detections.map((d) => d.label_name),
  };
}

export async function listImages(): Promise<ImageInfo[]> {
  const res = await authFetch("/images");
  if (!res.ok) throw new Error("Failed to fetch images");
  const data = await res.json();
  return data.images;
}

export function getImageUrl(imageId: number): string {
  return `${API_BASE}/images/${imageId}?t=${Date.now()}`;
}

export function getReportUrl(imageId: number): string {
  return `${API_BASE}/images/${imageId}/report?t=${Date.now()}`;
}

/* ---------- Stats API ---------- */

export interface UserStats {
  total_scans: number;
  total_detections: number;
  scans_this_month: number;
}

export async function fetchUserStats(): Promise<UserStats> {
  const res = await authFetch("/auth/stats");
  if (!res.ok) throw new Error("Failed to fetch stats");
  return res.json();
}

/* ---------- Case viewer (ticket 22) ---------- */

export interface CaseVolume {
  ncr: number;
  ed: number;
  et: number;
  total: number;
}

export interface CaseViewerCase {
  case_id: string;
  slices: number;
  volumes_cm3: CaseVolume;
}

export interface CaseSlice {
  slice_index: number;
  slices: number;
  base: string; // base64 PNG
  mask: string; // base64 PNG (RGBA overlay)
}

export async function listCaseViewerCases(): Promise<CaseViewerCase[]> {
  const res = await authFetch("/case-viewer/cases");
  if (!res.ok) throw new Error("Failed to fetch cases");
  const data = await res.json();
  return data.cases;
}

export async function getCaseViewerSlice(
  caseId: string,
  z: number,
): Promise<CaseSlice> {
  const res = await authFetch(`/case-viewer/cases/${caseId}/slices/${z}`);
  if (!res.ok) throw new Error("Failed to fetch slice");
  return res.json();
}

export async function getCaseViewerVolume(
  caseId: string,
  type: "t1ce" | "seg" | "pred",
): Promise<Blob> {
  const res = await authFetch(`/case-viewer/cases/${caseId}/volume?type=${type}`);
  if (!res.ok) throw new Error(`Failed to fetch ${type} volume`);
  return res.blob();
}

export async function segmentCaseViewerCase(
  caseId: string,
): Promise<{ seconds: number; volumes_cm3: Record<string, number> }> {
  const res = await authFetch(`/case-viewer/cases/${caseId}/segment`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Failed to segment case");
  return res.json();
}

/* ---------- Profile API ---------- */

export interface ProfileUpdate {
  username?: string;
  email?: string;
}

export async function updateProfile(data: ProfileUpdate): Promise<UserInfo> {
  const res = await authFetch("/auth/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Update failed" }));
    throw new Error(err.detail || "Update failed");
  }
  return res.json();
}

/* ---------- Password API ---------- */

export async function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<void> {
  const res = await authFetch("/auth/password", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Password change failed" }));
    throw new Error(err.detail || "Password change failed");
  }
}
