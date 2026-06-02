import React, { useEffect, useState } from "react";
import {
  Clock,
  AlertCircle,
  Image as ImageIcon,
  Trash2,
  FileText,
  Search,
  Filter,
  ArrowUpDown,
  Upload,
} from "lucide-react";
import {
  listImages,
  getImageUrl,
  getReportUrl,
  authFetch,
  detectionsToPredictionResult,
  type ImageInfo,
  type PredictionResult,
} from "../lib/api";
import type { DashboardPage } from "./Sidebar";

interface HistoryProps {
  onNavigate: (page: DashboardPage) => void;
  onViewImage: (imageId: number, results: PredictionResult | null) => void;
}

/* ─── Scan type distribution for badges ─── */
const SCAN_TYPES = [
  { pattern: "t1", badge: "neuro-t1", label: "NEURO_T1" },
  { pattern: "t2", badge: "neuro-t2", label: "NEURO_T2" },
  { pattern: "flair", badge: "flair", label: "FLAIR" },
  { pattern: "diff", badge: "diffusion", label: "DIFFUSION" },
  { pattern: "contrast", badge: "contrast", label: "CONTRAST" },
] as const;

function getScanBadge(filename: string): { badge: string; label: string } | null {
  const lower = filename.toLowerCase();
  for (const st of SCAN_TYPES) {
    if (lower.includes(st.pattern)) return { badge: st.badge, label: st.label };
  }
  // Default based on id hash
  return null;
}

export function History({ onNavigate, onViewImage }: HistoryProps) {
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [thumbnails, setThumbnails] = useState<Record<number, string>>({});

  const fetchImages = () => {
    setLoading(true);
    setError("");
    listImages()
      .then(setImages)
      .catch((err) => setError(err.message || "Failed to load images"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchImages(); }, []);

  // Fetch thumbnails via authFetch
  useEffect(() => {
    if (images.length === 0) return;
    for (const img of images) {
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
  }, [images]);

  const handleDelete = async (id: number) => {
    if (!window.confirm("Delete this scan? This action cannot be undone.")) return;
    setDeletingId(id);
    try {
      const res = await authFetch(`/images/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Delete failed");
      setImages((prev) => prev.filter((img) => img.id !== id));
    } catch (err: any) {
      setError(err.message || "Failed to delete image");
    } finally {
      setDeletingId(null);
    }
  };

  const handleDownloadReport = async (id: number) => {
    const url = getReportUrl(id);
    const resp = await authFetch(url);
    if (!resp.ok) { setError("Failed to generate report"); return; }
    const blob = await resp.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `report-${id}.pdf`;
    link.click();
  };

  const filtered = images.filter((img) =>
    img.original_name.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title" style={{ marginBottom: "0.25rem" }}>Scan History</h1>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            Review and manage clinical diagnostic records.
          </p>
        </div>
        {!loading && images.length > 0 && (
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <div className="search-bar">
              <Search size={16} />
              <input
                type="text"
                placeholder="Search by filename..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="search-input"
              />
            </div>
            <button className="btn btn-ghost" style={{ width: "auto", padding: "0.5rem 0.75rem" }} title="Filter">
              <Filter size={16} />
            </button>
            <button className="btn btn-ghost" style={{ width: "auto", padding: "0.5rem 0.75rem" }} title="Sort">
              <ArrowUpDown size={16} />
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="auth-error" style={{ marginBottom: "1rem" }}>
          <AlertCircle size={18} />
          <span>{error}</span>
          <button className="btn btn-ghost" onClick={fetchImages} style={{ marginLeft: "auto", width: "auto", padding: "0.3rem 0.8rem" }}>
            Retry
          </button>
        </div>
      )}

      {loading && (
        <div className="loading-state" style={{ padding: "3rem" }}>
          <div className="spinner" />
          <p>Loading history&hellip;</p>
        </div>
      )}

      {!loading && !error && filtered.length === 0 && (
        <div className="empty-state">
          <ImageIcon size={48} />
          <p>{search ? "No images match your search." : "No scans yet. Upload an MRI to get started."}</p>
          {!search && (
            <button className="btn btn-primary" onClick={() => onNavigate("detector")}>
              <Upload size={16} /> Upload Your First Scan
            </button>
          )}
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div className="history-list">
          {filtered.map((img) => {
            const scanBadge = getScanBadge(img.original_name);
            return (
              <div
                key={img.id}
                className="history-list-item"
                onClick={() => onViewImage(img.id, img.detections.length > 0 ? detectionsToPredictionResult(img.detections) : null)}
              >
                <img
                  src={thumbnails[img.id] || getImageUrl(img.id)}
                  alt={img.original_name}
                  className="history-list-thumb"
                  loading="lazy"
                />
                <div className="history-list-body">
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <span className="history-list-name">{img.original_name}</span>
                    {scanBadge && (
                      <span className={`scan-badge ${scanBadge.badge}`}>{scanBadge.label}</span>
                    )}
                  </div>
                  <div className="history-list-meta">
                    <span>{new Date(img.uploaded_at).toLocaleDateString()}</span>
                    <span style={{ color: "var(--text-secondary)" }}>•</span>
                    <span>{img.detections.length} detections</span>
                  </div>
                </div>
                <div className="history-list-actions">
                  <button
                    className="btn btn-tiny"
                    onClick={(e) => { e.stopPropagation(); handleDownloadReport(img.id); }}
                    title="Download Report"
                  >
                    <FileText size={14} />
                  </button>
                  <button
                    className="btn btn-tiny btn-danger"
                    onClick={(e) => { e.stopPropagation(); handleDelete(img.id); }}
                    disabled={deletingId === img.id}
                    title="Delete"
                  >
                    {deletingId === img.id ? <span className="spinner-small" /> : <Trash2 size={14} />}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
