import React, { useEffect, useState } from "react";
import {
  Scan,
  AlertCircle,
  Brain,
  CalendarDays,
  ArrowRight,
  Image as ImageIcon,
} from "lucide-react";
import {
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

export function DashboardHome({ onNavigate }: DashboardHomeProps) {
  const [stats, setStats] = useState<UserStats | null>(null);
  const [recentImages, setRecentImages] = useState<ImageInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([fetchUserStats(), listImages()])
      .then(([s, imgs]) => {
        setStats(s);
        setRecentImages(imgs.slice(0, 5));
      })
      .catch((err) => setError(err.message || "Failed to load dashboard"))
      .finally(() => setLoading(false));
  }, []);

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

      {/* Stat Cards */}
      <div className="stats-grid">
        <div className="dashboard-stat-card">
          <Scan size={24} className="stat-icon" />
          <div>
            <p className="stat-label">Total Scans</p>
            <p className="stat-value">{stats?.total_scans ?? 0}</p>
          </div>
        </div>
        <div className="dashboard-stat-card">
          <Brain size={24} className="stat-icon" />
          <div>
            <p className="stat-label">Detections Found</p>
            <p className="stat-value">{stats?.total_detections ?? 0}</p>
          </div>
        </div>
        <div className="dashboard-stat-card">
          <CalendarDays size={24} className="stat-icon" />
          <div>
            <p className="stat-label">Scans This Month</p>
            <p className="stat-value">{stats?.scans_this_month ?? 0}</p>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="section">
        <h2 className="section-title">Quick Actions</h2>
        <div className="quick-actions">
          <button className="action-card" onClick={() => onNavigate("detector")}>
            <Scan size={32} />
            <span>Upload New Scan</span>
            <ArrowRight size={18} />
          </button>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="section">
        <h2 className="section-title">Recent Activity</h2>
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
              >
                <img
                  src={getImageUrl(img.id)}
                  alt={img.original_name}
                  className="recent-thumb"
                />
                <div className="recent-info">
                  <span className="recent-name">{img.original_name}</span>
                  <span className="recent-date">
                    {new Date(img.uploaded_at).toLocaleDateString()}
                  </span>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
