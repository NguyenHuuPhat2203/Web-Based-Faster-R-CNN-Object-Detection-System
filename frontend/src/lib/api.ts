import { API_BASE, authFetch, setTokens } from "./session";
import { downloadBlob } from "./download";

/** Message of a caught value, for surfacing in the UI. */
export function messageOf(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}

/** Human-readable message from a FastAPI error body (`detail` as string or validation list). */
async function errorDetail(res: Response, fallback: string): Promise<string> {
  const body = await res.json().catch(() => null);
  const detail = body?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const msg = detail
      .map((d: { msg?: string }) => d.msg)
      .filter(Boolean)
      .join("; ");
    if (msg) return msg;
  }
  return fallback;
}

/* ---------- Auth API calls ---------- */

export interface UserInfo {
  id: number;
  email: string;
  username: string;
  created_at: string;
}

export interface LoginResponse extends UserInfo {
  access_token: string;
  refresh_token: string;
}

async function submitCredentials(
  path: string,
  body: Record<string, string>,
  fallback: string,
): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await errorDetail(res, fallback));
  const data: LoginResponse = await res.json();
  setTokens(data.access_token, data.refresh_token);
  return data;
}

export function register(
  email: string,
  username: string,
  password: string,
): Promise<LoginResponse> {
  return submitCredentials(
    "/auth/register",
    { email, username, password },
    "Registration failed",
  );
}

export function login(email: string, password: string): Promise<LoginResponse> {
  return submitCredentials("/auth/login", { email, password }, "Login failed");
}

/** Begin the Google OAuth flow: remember the CSRF state, then leave for Google. */
export async function startGoogleLogin(): Promise<void> {
  const res = await fetch(`${API_BASE}/auth/google`, { credentials: "include" });
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to initiate Google login"));
  const data: { redirect_url: string; state: string } = await res.json();
  sessionStorage.setItem("oauth_state", data.state);
  window.location.href = data.redirect_url;
}

export async function fetchCurrentUser(): Promise<UserInfo> {
  const res = await authFetch("/auth/me");
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to load profile"));
  return res.json();
}

/* ---------- Predict API ---------- */

export interface PredictionResult {
  /** xyxy in original image pixels. */
  boxes: number[][];
  labels: number[];
  scores: number[];
  label_names: string[];
  heatmap?: string | null; // base64 PNG, only for Faster R-CNN
  model_type?: string | null;
  model_version?: string | null;
  image_id?: number | null;
}

export type ModelType = "faster-rcnn" | "yolov8";

type ApiVersion = "v1" | "v2";

/** Backend API contract version. Default v2; set VITE_API_VERSION=v1 to rollback. */
const API_VERSION: ApiVersion =
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

  // The server reads threshold only from the query string, on every endpoint.
  const params = new URLSearchParams({ threshold: String(threshold) });
  let path: string;
  if (API_VERSION === "v2") {
    if (model === "faster-rcnn") {
      params.set("explain", options.explain ? "true" : "false");
      path = "/v2/predict";
    } else {
      path = "/v2/predict/yolo";
    }
  } else {
    path = model === "yolov8" ? "/predict-yolo" : "/predict";
  }
  const endpoint = `${path}?${params.toString()}`;

  const res = await authFetch(endpoint, { method: "POST", body: formData });

  if (!res.ok) {
    throw formatPredictError(await errorDetail(res, "Prediction failed"), res);
  }

  const data = await res.json();
  return API_VERSION === "v2"
    ? unwrapV2Prediction(data, res)
    : (data as PredictionResult);
}

/* ---------- Images API ---------- */

export interface DetectionInfo {
  label: number;
  label_name: string;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

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

/**
 * Detections the user's threshold confirmed. The server stores everything down to
 * its 0.1 floor, so lists, counts and reports must apply the image's own threshold.
 */
export function confirmedDetections(image: ImageInfo): DetectionInfo[] {
  const min = image.threshold ?? 0;
  return image.detections.filter((d) => d.score >= min);
}

/** Stored image detections in the PredictionResult shape used by the viewers. */
export function imageToPredictionResult(image: ImageInfo): PredictionResult {
  const detections = confirmedDetections(image);
  return {
    boxes: detections.map((d) => [d.x1, d.y1, d.x2, d.y2]),
    labels: detections.map((d) => d.label),
    scores: detections.map((d) => d.score),
    label_names: detections.map((d) => d.label_name),
  };
}

export async function listImages(): Promise<ImageInfo[]> {
  const res = await authFetch("/images");
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to fetch images"));
  const data = await res.json();
  return data.images;
}

/** Image file as a blob; protected, so it cannot be used directly as an <img src>. */
export async function fetchImageBlob(imageId: number): Promise<Blob> {
  const res = await authFetch(`/images/${imageId}`);
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to load image"));
  return res.blob();
}

export async function deleteImage(imageId: number): Promise<void> {
  const res = await authFetch(`/images/${imageId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to delete image"));
}

/** Fetch the PDF report with the Bearer token and hand it to the browser as a download. */
export async function downloadReport(imageId: number): Promise<void> {
  const res = await authFetch(`/images/${imageId}/report`);
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to generate report"));
  downloadBlob(await res.blob(), `report-${imageId}.pdf`);
}

/* ---------- Stats API ---------- */

export interface UserStats {
  total_scans: number;
  total_detections: number;
  scans_this_month: number;
}

export async function fetchUserStats(): Promise<UserStats> {
  const res = await authFetch("/auth/stats");
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to fetch stats"));
  return res.json();
}

/* ---------- Case viewer ---------- */

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

export interface SegmentationResult {
  seconds: number;
  volumes_cm3: CaseVolume;
}

export type CaseVolumeType = "t1ce" | "seg" | "pred";

export async function listCaseViewerCases(): Promise<CaseViewerCase[]> {
  const res = await authFetch("/case-viewer/cases");
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to fetch cases"));
  const data = await res.json();
  return data.cases;
}

export async function getCaseViewerVolume(
  caseId: string,
  type: CaseVolumeType,
): Promise<Blob> {
  const res = await authFetch(
    `/case-viewer/cases/${encodeURIComponent(caseId)}/volume?type=${type}`,
  );
  if (!res.ok) throw new Error(await errorDetail(res, `Failed to fetch ${type} volume`));
  return res.blob();
}

/** Run model segmentation; the server reports region volumes keyed NCR/ED/ET. */
export async function segmentCaseViewerCase(caseId: string): Promise<SegmentationResult> {
  const res = await authFetch(`/case-viewer/cases/${encodeURIComponent(caseId)}/segment`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await errorDetail(res, "Failed to segment case"));
  const data: { seconds: number; volumes_cm3: Record<string, number> } = await res.json();
  const v = data.volumes_cm3;
  const ncr = v.NCR ?? v.ncr ?? 0;
  const ed = v.ED ?? v.ed ?? 0;
  const et = v.ET ?? v.et ?? 0;
  return { seconds: data.seconds, volumes_cm3: { ncr, ed, et, total: ncr + ed + et } };
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
  if (!res.ok) throw new Error(await errorDetail(res, "Update failed"));
  return res.json();
}

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
  if (!res.ok) throw new Error(await errorDetail(res, "Password change failed"));
}
