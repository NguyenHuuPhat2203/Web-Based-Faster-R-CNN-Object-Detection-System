import React, { useEffect } from "react";
import { useAuth } from "../context/AuthContext";

export function OAuthCallback() {
  const { handleOAuthCallback } = useAuth();

  useEffect(() => {
    handleOAuthCallback();
    // Clean the URL after processing tokens
    window.history.replaceState(null, "", "/");
  }, [handleOAuthCallback]);

  return (
    <div style={{ textAlign: "center", padding: "4rem", color: "#94a3b8" }}>
      <div className="spinner" />
      <p>Completing sign in…</p>
    </div>
  );
}
