import { useCallback, useEffect, useState } from "react";
import {
  Scan,
  AlertCircle,
  Brain,
  CalendarDays,
  ArrowRight,
  Image as ImageIcon,
  CheckCircle,
} from "lucide-react";
import { useNavigate, useOutletContext } from "react-router-dom";
import {
  messageOf,
  fetchUserStats,
  listImages,
  confirmedDetections,
  imageToPredictionResult,
  type UserStats,
  type ImageInfo,
} from "../lib/api";
import { isTumor } from "../lib/detections";
import { useImageBlobUrls } from "../lib/useImageBlobUrls";
import type { LayoutOutletContext } from "./DashboardLayout";

export function DashboardHome() {
  const navigate = useNavigate();
  const { onViewImage } = useOutletContext<LayoutOutletContext>();
  const [stats, setStats] = useState<UserStats | null>(null);
  const [recentImages, setRecentImages] = useState<ImageInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    setLoading(true);
    setError("");
    Promise.all([fetchUserStats(), listImages()])
      .then(([s, imgs]) => {
        setStats(s);
        setRecentImages(imgs.slice(0, 5));
      })
      .catch((err) => setError(messageOf(err, "Failed to load dashboard")))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  const thumbnails = useImageBlobUrls(recentImages.map((img) => img.id));

  const getStatusBadge = (img: ImageInfo) =>
    confirmedDetections(img).some((d) => isTumor(d.label_name)) ? (
      <span className="status-badge detection"><AlertCircle size={10} /> Detection</span>
    ) : (
      <span className="status-badge completed"><CheckCircle size={10} /> Completed</span>
    );

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
          <button className="btn btn-secondary" onClick={load}>
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

      {/* Quick Actions */}
      <div style={{ marginBottom: "2rem" }}>
        {/* Quick Actions */}
        <div className="section" style={{ marginBottom: 0 }}>
          <h2 className="section-title">Quick Actions</h2>
          <div className="quick-actions">
            <button className="action-card" onClick={() => navigate("/detector")}>
              <Scan size={28} />
              <span>Upload New Scan</span>
              <p className="text-muted" style={{ fontSize: "0.78rem", marginTop: "0.25rem" }}>
                Start new MRI analysis
              </p>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
          <h2 className="section-title" style={{ marginBottom: 0 }}>Recent Activity</h2>
          <button className="btn btn-ghost" style={{ width: "auto", padding: "0.35rem 0.8rem", fontSize: "0.8rem" }} onClick={() => navigate("/history")}>
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
                onClick={() => onViewImage(img.id, imageToPredictionResult(img))}
                style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}
              >
                {thumbnails[img.id] ? (
                  <img src={thumbnails[img.id]} alt={img.original_name} className="recent-thumb" />
                ) : (
                  <div className="recent-thumb" />
                )}
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