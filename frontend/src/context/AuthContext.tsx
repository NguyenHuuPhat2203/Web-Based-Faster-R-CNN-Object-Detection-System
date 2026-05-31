import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import {
  isAuthenticated,
  login as apiLogin,
  register as apiRegister,
  logout as apiLogout,
  processOAuthTokens,
  refreshTokens,
  type UserInfo,
} from "../lib/api";

/* ---------- Types ---------- */

interface AuthState {
  isAuth: boolean;
  loading: boolean; // true while we check / refresh on mount
}

interface AuthContextValue extends AuthState {
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

  // On mount: if we have a token, try to refresh it (validates it's still good)
  useEffect(() => {
    if (isAuthenticated()) {
      refreshTokens()
        .then((ok) => {
          setState({ isAuth: ok, loading: false });
          if (!ok) apiLogout();
        })
        .catch(() => setState({ isAuth: false, loading: false }));
    } else {
      setState({ isAuth: false, loading: false });
    }
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
    processOAuthTokens();
    setState({ isAuth: isAuthenticated(), loading: false });
  }, []);

  return (
    <AuthContext.Provider
      value={{ ...state, login, register, logout, handleOAuthCallback }}
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
