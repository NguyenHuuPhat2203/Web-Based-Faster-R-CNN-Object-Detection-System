// Session and token-aware transport: token storage, refresh, and authFetch.
// Endpoint helpers live in api.ts and import from here.

export const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export function getAccessToken(): string | null {
  return localStorage.getItem("access_token");
}

export function getRefreshToken(): string | null {
  return localStorage.getItem("refresh_token");
}

export function setTokens(access: string, refresh: string) {
  localStorage.setItem("access_token", access);
  localStorage.setItem("refresh_token", refresh);
}

export function clearTokens() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
}

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
    } else {
      // Refresh failed → session is dead: clear tokens and let the auth gate
      // redirect to /login (RequireAuth picks this up via the event).
      clearTokens();
      window.dispatchEvent(new Event("auth:expired"));
    }
  }

  return res;
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
