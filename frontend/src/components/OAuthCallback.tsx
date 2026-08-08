import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function OAuthCallback() {
  const { handleOAuthCallback } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    handleOAuthCallback();
    navigate("/", { replace: true });
  }, [handleOAuthCallback, navigate]);

  return (
    <div style={{ textAlign: "center", padding: "4rem", color: "#94a3b8" }}>
      <div className="spinner" />
      <p>Completing sign in…</p>
    </div>
  );
}
