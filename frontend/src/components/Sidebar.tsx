import { useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Scan,
  Clock,
  Settings,
  LogOut,
  Brain,
  ChevronLeft,
  Menu,
  Boxes,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";

export function Sidebar() {
  const { logout, user } = useAuth();
  const [collapsed, setCollapsed] = useState(false);

  const items: { to: string; end?: boolean; label: string; icon: ReactNode }[] = [
    { to: "/", end: true, label: "Dashboard", icon: <LayoutDashboard size={19} /> },
    { to: "/detector", label: "Detector", icon: <Scan size={19} /> },
    { to: "/history", label: "History", icon: <Clock size={19} /> },
    { to: "/case-viewer", label: "Case Viewer", icon: <Boxes size={19} /> },
    { to: "/settings", label: "Settings", icon: <Settings size={19} /> },
  ];

  const initials = (user?.username?.[0] ?? "?").toUpperCase();

  return (
    <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
      {/* Branding */}
      <div className="sidebar-header">
        <Brain size={24} style={{ color: "var(--accent)" }} />
        {!collapsed && (
          <div>
            <span className="sidebar-brand">NeuroScan AI</span>
            <span className="sidebar-brand-sub">Clinical Portal</span>
          </div>
        )}
        <button className="sidebar-collapse-btn" onClick={() => setCollapsed(!collapsed)}>
          {collapsed ? <Menu size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>

      {/* User avatar */}
      {!collapsed && (
        <div className="sidebar-user">
          <div
            className="user-avatar"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "var(--primary-subtle)",
              color: "var(--primary)",
              fontWeight: 700,
              fontSize: "1rem",
            }}
          >
            {initials}
          </div>
          <div className="sidebar-user-info">
            <span className="sidebar-user-name">{user?.username ?? "…"}</span>
            <span className="sidebar-user-role">{user?.email ?? ""}</span>
          </div>
        </div>
      )}

      {/* Navigation */}
      <nav className="sidebar-nav">
        {items.map(({ to, end, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `sidebar-item${isActive ? " active" : ""}`}
          >
            {icon}
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Footer section */}
      <div className="sidebar-footer">
        <button className="sidebar-item logout" onClick={logout}>
          <LogOut size={19} />
          {!collapsed && <span>Sign Out</span>}
        </button>
      </div>

      {/* Status footer */}
      {!collapsed && (
        <div className="status-footer">
          <div className="status-footer-left">
            <span className="status-dot" />
            <span>System Online</span>
          </div>
        </div>
      )}
    </aside>
  );
}
