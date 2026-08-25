import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  authFetch,
  clearTokens,
  getAccessToken,
  refreshTokens,
  setTokens,
} from "./session";

const store = new Map<string, string>();

beforeEach(() => {
  store.clear();
  Object.defineProperty(globalThis, "localStorage", {
    value: {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => store.set(k, v),
      removeItem: (k: string) => store.delete(k),
    },
    configurable: true,
  });
  vi.stubGlobal("window", { dispatchEvent: vi.fn() });
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
});
