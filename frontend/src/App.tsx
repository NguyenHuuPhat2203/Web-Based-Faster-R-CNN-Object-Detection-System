import type { ReactNode } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useAuth } from "./context/AuthContext";
import { Login } from "./components/Login";
import { Register } from "./components/Register";
import { OAuthCallback } from "./components/OAuthCallback";
import { DashboardLayout } from "./components/DashboardLayout";
import { DashboardHome } from "./components/DashboardHome";
import { Detector } from "./components/Detector";
import { History } from "./components/History";
import { Settings } from "./components/Settings";

function LoadingScreen() {
  return (
    <div className="loading-screen" style={{ flexDirection: "column", gap: "1rem" }}>
      <div className="spinner" />
      <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>NeuroScan AI — Loading</p>
    </div>
  );
}

/** Redirect to /login when unauthenticated, remembering the intended destination. */
function RequireAuth({ children }: { children: ReactNode }) {
  const { isAuth, loading } = useAuth();
  const location = useLocation();

  if (loading) return <LoadingScreen />;
  if (!isAuth) {
    return (
      <Navigate
        to="/login"
        replace
        state={{ from: location.pathname + location.search }}
      />
    );
  }
  return <>{children}</>;
}

function App() {
  const { loading } = useAuth();
  if (loading) return <LoadingScreen />;

  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/oauth-callback" element={<OAuthCallback />} />

      <Route
        path="/"
        element={
          <RequireAuth>
            <DashboardLayout />
          </RequireAuth>
        }
      >
        <Route index element={<DashboardHome />} />
        <Route path="detector" element={<Detector />} />
        <Route path="history" element={<History />} />
        <Route path="settings" element={<Settings />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default App;
