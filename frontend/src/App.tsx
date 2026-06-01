import { useState, useEffect } from "react";
import { useAuth } from "./context/AuthContext";
import { Login } from "./components/Login";
import { Register } from "./components/Register";
import { OAuthCallback } from "./components/OAuthCallback";
import { DashboardLayout } from "./components/DashboardLayout";

type Page = "login" | "register" | "oauth-callback";

function App() {
  const { isAuth, loading } = useAuth();
  const [page, setPage] = useState<Page>(() => {
    const path = window.location.pathname;
    if (path.startsWith("/oauth-callback")) return "oauth-callback";
    return "login";
  });

  // Once authenticated, DashboardLayout handles all in-app navigation
  useEffect(() => {
    if (isAuth && page !== "oauth-callback") {
      setPage("login");
    }
  }, [isAuth, page]);

  if (loading) {
    return <div className="loading-screen"><div className="spinner" /></div>;
  }

  if (page === "oauth-callback") {
    return <OAuthCallback />;
  }

  if (!isAuth) {
    if (page === "register") {
      return <Register onSwitchToLogin={() => setPage("login")} />;
    }
    return <Login onSwitchToRegister={() => setPage("register")} />;
  }

  return <DashboardLayout />;
}

export default App;
