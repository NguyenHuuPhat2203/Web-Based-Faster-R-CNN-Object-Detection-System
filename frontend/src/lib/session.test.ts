import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  authFetch,
  clearTokens,
  getAccessToken,
  processOAuthTokens,
  refreshTokens,
  setTokens,
} from "./session";

const store = new Map<string, string>();
const session = new Map<string, string>();
const location = { hash: "", pathname: "/" };

beforeEach(() => {
  store.clear();
  session.clear();
  location.hash = "";
  Object.defineProperty(globalThis, "localStorage", {
    value: {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => store.set(k, v),
      removeItem: (k: string) => store.delete(k),
    },
    configurable: true,
  });
  Object.defineProperty(globalThis, "sessionStorage", {
    value: {
      getItem: (k: string) => session.get(k) ?? null,
      setItem: (k: string, v: string) => session.set(k, v),
      removeItem: (k: string) => session.delete(k),
      clear: () => session.clear(),
    },
    configurable: true,
  });
  vi.stubGlobal("window", {
    dispatchEvent: vi.fn(),
    location,
    history: { replaceState: vi.fn() },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("token storage", () => {
  it("round-trips access and refresh tokens", () => {
    setTokens("acc", "ref");
    expect(getAccessToken()).toBe("acc");
    clearTokens();
    expect(getAccessToken()).toBeNull();
  });
});

describe("refreshTokens", () => {
  it("returns false without a refresh token", async () => {
    expect(await refreshTokens()).toBe(false);
  });

  it("stores new tokens when the backend accepts the refresh", async () => {
    setTokens("old-acc", "old-ref");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ access_token: "new-acc", refresh_token: "new-ref" }),
      }),
    );
    expect(await refreshTokens()).toBe(true);
    expect(getAccessToken()).toBe("new-acc");
  });

  it("returns false when the backend rejects the refresh", async () => {
    setTokens("old-acc", "old-ref");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false }));
    expect(await refreshTokens()).toBe(false);
    expect(getAccessToken()).toBe("old-acc");
  });
});

describe("refreshTokens concurrency", () => {
  it("shares one request between concurrent callers", async () => {
    setTokens("old-acc", "old-ref");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ access_token: "new-acc", refresh_token: "new-ref" }),
    });
    vi.stubGlobal("fetch", fetchMock);
    const results = await Promise.all([refreshTokens(), refreshTokens()]);
    expect(results).toEqual([true, true]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("rejects on a server error instead of reporting a dead session", async () => {
    setTokens("old-acc", "old-ref");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    await expect(refreshTokens()).rejects.toThrow();
    expect(getAccessToken()).toBe("old-acc");
  });
});

describe("authFetch", () => {
  it("attaches the bearer token and retries once after a 401 refresh", async () => {
    setTokens("acc", "ref");
    const queue = [
      { status: 401, ok: false },
      {
        status: 200,
        ok: true,
        json: async () => ({ access_token: "new-acc", refresh_token: "new-ref" }),
      },
      { status: 200, ok: true },
    ];
    const seen: (string | undefined)[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn((_url: string, opts?: RequestInit) => {
        seen.push((opts?.headers as Record<string, string>)?.Authorization);
        return Promise.resolve(queue.shift());
      }),
    );

    const res = await authFetch("/images");

    expect(res.status).toBe(200);
    expect(seen).toEqual(["Bearer acc", undefined, "Bearer new-acc"]);
  });

  it("keeps the session when the refresh endpoint is unreachable", async () => {
    setTokens("acc", "ref");
    const queue = [{ status: 401, ok: false }];
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        queue.length ? Promise.resolve(queue.shift()) : Promise.reject(new TypeError("offline")),
      ),
    );
    const res = await authFetch("/images");
    expect(res.status).toBe(401);
    expect(getAccessToken()).toBe("acc");
  });
});

describe("processOAuthTokens", () => {
  it("rejects fragment tokens without matching oauth_state", () => {
    sessionStorage.clear();
    window.location.hash = "#access_token=a&refresh_token=b&state=nope";
    expect(processOAuthTokens()).toBe(false);
    expect(getAccessToken()).toBeNull();
  });

  it("accepts fragment tokens when state matches sessionStorage", () => {
    sessionStorage.setItem("oauth_state", "abc");
    window.location.hash = "#access_token=a&refresh_token=b&state=abc";
    expect(processOAuthTokens()).toBe(true);
    expect(getAccessToken()).toBe("a");
    expect(sessionStorage.getItem("oauth_state")).toBeNull();
  });
});
