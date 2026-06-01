// Use env var with fallback for configurable API URL
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/* ---------- Token helpers ---------- */

function getAccessToken(): string | null {
  return localStorage.getItem("access_token");
}

function getRefreshToken(): string | null {
  return localStorage.getItem("refresh_token");
}

function setTokens(access: string, refresh: string) {
  localStorage.setItem("access_token", access);
  localStorage.setItem("refresh_token", refresh);
}

function clearTokens() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}

/* ---------- Auth helpers ---------- */

export function isAuthenticated(): boolean {
  return !!getAccessToken();
}

export function logout(): void {
  const access = getAccessToken();
  const refresh = getRefreshToken();
  // Fire-and-forget logout to the backend (blacklist the tokens)
  if (access && refresh) {
    fetch(`${API_BASE}/auth/logout`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ access_token: access, refresh_token: refresh }),
    }).catch(() => {});
  }
  clearTokens();
}

export async function refreshTokens(): Promise<boolean> {
  const rt = getRefreshToken();
  if (!rt) return false;

  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: rt }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    setTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

/* ---------- Generic fetch with auth ---------- */

export async function authFetch(
  path: string,
  options: RequestInit = {},
): Promise<Response> {
  const token = getAccessToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  // Token expired → try refreshing once
  if (res.status === 401 && getRefreshToken()) {
    const ok = await refreshTokens();
    if (ok) {
      headers["Authorization"] = `Bearer ${getAccessToken()}`;
      res = await fetch(`${API_BASE}${path}`, { ...options, headers });
    }
  }

  return res;
}

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

export function processOAuthTokens() {
  const hash = window.location.hash.substring(1);
  const params = new URLSearchParams(hash);
  const access = params.get("access_token");
  const refresh = params.get("refresh_token");
  if (access && refresh) {
    setTokens(access, refresh);
  }
  // Clean the URL
  window.location.hash = "";
  window.history.replaceState(null, "", window.location.pathname);
}

/* ---------- Predict API ---------- */

export interface PredictionResult {
  boxes: number[][];
  labels: number[];
  scores: number[];
  label_names: string[];
  heatmap?: string | null;  // base64 PNG, only for Faster R-CNN
}

export type ModelType = "faster-rcnn" | "yolov8";

export async function predict(
  file: File,
  threshold = 0.5,
  model: ModelType = "faster-rcnn",
): Promise<PredictionResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("threshold", String(threshold));

  const endpoint = model === "yolov8" ? "/predict-yolo" : "/predict";

  const res = await authFetch(endpoint, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Prediction failed" }));
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

/* ---------- Images API ---------- */

export interface ImageInfo {
  id: number;
  original_name: string;
  mime_type: string;
  detection_result: PredictionResult | null;
  uploaded_at: string;
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
