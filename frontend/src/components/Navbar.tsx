import React from "react";
import { LogOut, Brain, User } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export function Navbar() {
  const { isAuth, logout } = useAuth();

  if (!isAuth) return null;

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Brain size={24} />
        <span>Brain Tumor Detector</span>
      </div>

      <div className="navbar-actions">
        <button className="btn btn-ghost" onClick={logout}>
          <LogOut size={18} />
          Sign Out
        </button>
      </div>
    </nav>
  );
}
