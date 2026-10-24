import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import {
  isAuthenticated,
  logout as apiLogout,
  processOAuthTokens,
  refreshTokens,
} from "../lib/session";
import {
  fetchCurrentUser,
  login as apiLogin,
  register as apiRegister,
  type UserInfo,
} from "../lib/api";

/* ---------- Types ---------- */

interface AuthState {
  isAuth: boolean;
  loading: boolean; // true while we check / refresh on mount
}

interface AuthContextValue extends AuthState {
  user: UserInfo | null;
  setUser: (user: UserInfo) => void;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, username: string, password: string) => Promise<void>;
  logout: () => void;
  handleOAuthCallback: () => void;
}

/* ---------- Context ---------- */

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    isAuth: false,
    loading: true,
  });
  const [user, setUser] = useState<UserInfo | null>(null);

  // On mount: if we have a token, try to refresh it (validates it's still good)
  useEffect(() => {
    if (!isAuthenticated()) {
      setState({ isAuth: false, loading: false });
      return;
    }
    refreshTokens()
      .then((ok) => {
        if (!ok) apiLogout();
        setState({ isAuth: ok, loading: false });
      })
      // Server unreachable: the session is not proven dead, keep the tokens.
      .catch(() => setState({ isAuth: true, loading: false }));
  }, []);

  // Load the signed-in user's profile whenever a session becomes active.
  useEffect(() => {
    if (!state.isAuth) {
      setUser(null);
      return;
    }
    let cancelled = false;
    fetchCurrentUser()
      .then((u) => {
        if (!cancelled) setUser(u);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [state.isAuth]);

  // Session died mid-request (refresh failed in authFetch) → force logout state
  // so RequireAuth redirects to /login.
  useEffect(() => {
    const onExpired = () => setState({ isAuth: false, loading: false });
    window.addEventListener("auth:expired", onExpired);
    return () => window.removeEventListener("auth:expired", onExpired);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    await apiLogin(email, password);
    setState({ isAuth: true, loading: false });
  }, []);

  const register = useCallback(
    async (email: string, username: string, password: string) => {
      await apiRegister(email, username, password);
      setState({ isAuth: true, loading: false });
    },
    [],
  );

  const logout = useCallback(() => {
    apiLogout();
    setState({ isAuth: false, loading: false });
  }, []);

  const handleOAuthCallback = useCallback(() => {
    // Rejected callbacks (bad state, or a StrictMode re-run after the fragment
    // was consumed) must not clobber an already-established session.
    if (processOAuthTokens()) setState({ isAuth: true, loading: false });
  }, []);

  return (
    <AuthContext.Provider
      value={{ ...state, user, setUser, login, register, logout, handleOAuthCallback }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}