import React from "react";
import {
  LayoutDashboard,
  Scan,
  Clock,
  Settings,
  LogOut,
  Brain,
  ChevronLeft,
  Menu,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";

export type DashboardPage = "home" | "detector" | "history" | "settings";

interface SidebarProps {
  activePage: DashboardPage;
  onNavigate: (page: DashboardPage) => void;
}

export function Sidebar({ activePage, onNavigate }: SidebarProps) {
  const { logout } = useAuth();
  const [collapsed, setCollapsed] = React.useState(false);

  const items: { page: DashboardPage; label: string; icon: React.ReactNode }[] = [
    { page: "home", label: "Dashboard", icon: <LayoutDashboard size={20} /> },
    { page: "detector", label: "Detector", icon: <Scan size={20} /> },
    { page: "history", label: "History", icon: <Clock size={20} /> },
    { page: "settings", label: "Settings", icon: <Settings size={20} /> },
  ];

  return (
    <aside className={`sidebar${collapsed ? " collapsed" : ""}`}>
      <div className="sidebar-header">
        <Brain size={28} />
        {!collapsed && <span className="sidebar-brand">Brain Tumor<br />Detector</span>}
        <button className="sidebar-collapse-btn" onClick={() => setCollapsed(!collapsed)}>
          {collapsed ? <Menu size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      <nav className="sidebar-nav">
        {items.map(({ page, label, icon }) => (
          <button
            key={page}
            className={`sidebar-item${activePage === page ? " active" : ""}`}
            onClick={() => onNavigate(page)}
          >
            {icon}
            {!collapsed && <span>{label}</span>}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button className="sidebar-item logout" onClick={logout}>
          <LogOut size={20} />
          {!collapsed && <span>Sign Out</span>}
        </button>
      </div>
    </aside>
  );
}
