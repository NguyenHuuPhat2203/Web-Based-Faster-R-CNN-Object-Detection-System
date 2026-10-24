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

let refreshInFlight: Promise<boolean> | null = null;

async function requestRefresh(): Promise<boolean> {
  const rt = getRefreshToken();
  if (!rt) return false;

  const res = await fetch(`${API_BASE}/auth/refresh`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: rt }),
  });
  // Server outage is not a dead session: surface it instead of logging out.
  if (res.status >= 500) throw new Error(`Refresh failed (${res.status})`);
  if (!res.ok) return false;
  const data = await res.json();
  setTokens(data.access_token, data.refresh_token);
  return true;
}

/**
 * Refresh the token pair. Concurrent callers share one request: refresh tokens
 * are single-use, so parallel refreshes would invalidate each other.
 * Resolves false when the session is rejected; rejects on network/server errors.
 */
export function refreshTokens(): Promise<boolean> {
  refreshInFlight ??= requestRefresh().finally(() => {
    refreshInFlight = null;
  });
  return refreshInFlight;
}

export async function authFetch(
  path: string,
  options: RequestInit = {},
): Promise<Response> {
  const sentToken = getAccessToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (sentToken) headers["Authorization"] = `Bearer ${sentToken}`;

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (res.status !== 401 || !getRefreshToken()) return res;

  // Another request may already have rotated the tokens while this one was in flight.
  let renewed = getAccessToken() !== sentToken;
  if (!renewed) {
    try {
      renewed = await refreshTokens();
    } catch {
      return res; // transient failure: keep the session, surface the 401
    }
    if (!renewed) {
      // Session is dead: clear tokens and let the auth gate redirect to /login.
      clearTokens();
      window.dispatchEvent(new Event("auth:expired"));
      return res;
    }
  }
  headers["Authorization"] = `Bearer ${getAccessToken()}`;
  return fetch(`${API_BASE}${path}`, { ...options, headers });
}

export function processOAuthTokens(): boolean {
  const hash = window.location.hash.substring(1);
  const params = new URLSearchParams(hash);
  const access = params.get("access_token");
  const refresh = params.get("refresh_token");
  const state = params.get("state");
  const expected = sessionStorage.getItem("oauth_state");
  const accepted = Boolean(access && refresh && expected && expected === state);
  if (accepted) {
    setTokens(access as string, refresh as string);
    sessionStorage.removeItem("oauth_state");
  }
  window.location.hash = "";
  window.history.replaceState(null, "", window.location.pathname);
  return accepted;
}
