import React, { useEffect, useState } from "react";
import {
  Scan,
  AlertCircle,
  Brain,
  CalendarDays,
  ArrowRight,
  Image as ImageIcon,
  Activity,
  CheckCircle,
  Clock,
} from "lucide-react";
import {
  authFetch,
  fetchUserStats,
  listImages,
  getImageUrl,
  detectionsToPredictionResult,
  type UserStats,
  type ImageInfo,
  type PredictionResult,
} from "../lib/api";
import type { DashboardPage } from "./Sidebar";

interface DashboardHomeProps {
  onNavigate: (page: DashboardPage) => void;
  onViewImage: (imageId: number, results: PredictionResult | null) => void;
}

/* ─── Chart data ─── */
const MONTHLY_ACCURACY = [
  { month: "Jan", value: 97.5 },
  { month: "Feb", value: 97.8 },
  { month: "Mar", value: 98.2 },
  { month: "Apr", value: 99.4 },
  { month: "May", value: 99.1 },
  { month: "Jun", value: 99.2 },
];

export function DashboardHome({ onNavigate, onViewImage }: DashboardHomeProps) {
  const [stats, setStats] = useState<UserStats | null>(null);
  const [recentImages, setRecentImages] = useState<ImageInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [thumbnails, setThumbnails] = useState<Record<number, string>>({});

  useEffect(() => {
    Promise.all([fetchUserStats(), listImages()])
      .then(([s, imgs]) => {
        setStats(s);
        setRecentImages(imgs.slice(0, 5));
      })
      .catch((err) => setError(err.message || "Failed to load dashboard"))
      .finally(() => setLoading(false));
  }, []);

  // Fetch thumbnails via authFetch
  useEffect(() => {
    if (recentImages.length === 0) return;
    for (const img of recentImages) {
      if (thumbnails[img.id]) continue;
      authFetch(`/images/${img.id}`)
        .then((r) => (r.ok ? r.blob() : null))
        .then((blob) => {
          if (blob) {
            const url = URL.createObjectURL(blob);
            setThumbnails((prev) => ({ ...prev, [img.id]: url }));
          }
        })
        .catch(() => {});
    }
  }, [recentImages]);

  const getStatusBadge = (img: ImageInfo) => {
    if (img.detections.length > 0) {
      const hasTumor = img.detections.some((d: any) => d.confidence >= 0.3);
      if (hasTumor) {
        return <span className="status-badge detection"><AlertCircle size={10} /> Detection</span>;
      }
      return <span className="status-badge completed"><CheckCircle size={10} /> Completed</span>;
    }
    return <span className="status-badge processing"><Clock size={10} /> Processing</span>;
  };

  const chartHeight = 120;
  const maxVal = Math.max(...MONTHLY_ACCURACY.map(d => d.value));
  const minVal = 95;

  if (loading) {
    return (
      <div className="page-container">
        <div className="loading-state"><div className="spinner" /><p>Loading dashboard&hellip;</p></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page-container">
        <div className="error-state">
          <AlertCircle size={24} />
          <p>{error}</p>
          <button className="btn btn-secondary" onClick={() => window.location.reload()}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page-container">
      <h1 className="page-title">Dashboard</h1>

      {/* KPI Cards */}
      <div className="stats-grid">
        <div className="dashboard-stat-card">
          <Scan size={22} className="stat-icon" />
          <div>
            <p className="stat-label">Total Scans</p>
            <p className="stat-value">{stats?.total_scans ?? 0}</p>
          </div>
        </div>
        <div className="dashboard-stat-card">
          <Brain size={22} className="stat-icon" />
          <div>
            <p className="stat-label">Detections Found</p>
            <p className="stat-value">{stats?.total_detections ?? 0}</p>
          </div>
        </div>
        <div className="dashboard-stat-card">
          <CalendarDays size={22} className="stat-icon" />
          <div>
            <p className="stat-label">Scans This Month</p>
            <p className="stat-value">{stats?.scans_this_month ?? 0}</p>
          </div>
        </div>
      </div>

      {/* Quick Actions + Chart row */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "2rem" }}>
        {/* Quick Actions */}
        <div className="section" style={{ marginBottom: 0 }}>
          <h2 className="section-title">Quick Actions</h2>
          <div className="quick-actions">
            <button className="action-card" onClick={() => onNavigate("detector")}>
              <Scan size={28} />
              <span>Upload New Scan</span>
              <p className="text-muted" style={{ fontSize: "0.78rem", marginTop: "0.25rem" }}>
                Start new MRI analysis
              </p>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>

        {/* Accuracy Chart */}
        <div className="chart-card">
          <div className="chart-header">
            <div>
              <div className="chart-title">Analysis Accuracy</div>
              <div className="chart-subtitle">Monthly %</div>
            </div>
            <div className="chart-legend">
              <span className="chart-legend-dot" />
              AI Model v4.2 — 99.2% Accuracy
            </div>
          </div>
          <div className="chart-bars">
            {MONTHLY_ACCURACY.map((d) => {
              const pct = ((d.value - minVal) / (maxVal - minVal)) * 90 + 10;
              const isHighest = d.value >= 99.4;
              return (
                <div key={d.month} className="chart-bar-group">
                  <div
                    className="chart-bar"
                    style={{
                      height: `${pct}%`,
                      background: isHighest
                        ? "linear-gradient(to top, var(--primary), #22d3ee)"
                        : "linear-gradient(to top, rgba(6, 182, 212, 0.4), rgba(6, 182, 212, 0.15))",
                    }}
                  />
                  <span className="chart-bar-value">{d.value}%</span>
                  <span className="chart-bar-label">{d.month}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
          <h2 className="section-title" style={{ marginBottom: 0 }}>Recent Activity</h2>
          <button className="btn btn-ghost" style={{ width: "auto", padding: "0.35rem 0.8rem", fontSize: "0.8rem" }} onClick={() => onNavigate("history")}>
            View All <ArrowRight size={14} />
          </button>
        </div>
        {recentImages.length === 0 ? (
          <div className="empty-state">
            <ImageIcon size={32} />
            <p>No scans yet. Upload an MRI to get started.</p>
          </div>
        ) : (
          <div className="recent-list">
            {recentImages.map((img) => (
              <button
                key={img.id}
                className="recent-item"
                onClick={() => onViewImage(img.id, img.detections.length > 0 ? detectionsToPredictionResult(img.detections) : null)}
                style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}
              >
                <img
                  src={thumbnails[img.id] || getImageUrl(img.id)}
                  alt={img.original_name}
                  className="recent-thumb"
                />
                <div className="recent-info" style={{ flex: 1 }}>
                  <span className="recent-name">{img.original_name}</span>
                  <span className="recent-date">
                    {new Date(img.uploaded_at).toLocaleDateString()}
                  </span>
                </div>
                {getStatusBadge(img)}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
