import React, { useEffect, useState } from "react";
import {
  Clock,
  AlertCircle,
  Image as ImageIcon,
  Trash2,
  FileText,
  Search,
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

export function History({ onNavigate }: HistoryProps) {
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const fetchImages = () => {
    setLoading(true);
    setError("");
    listImages()
      .then(setImages)
      .catch((err) => setError(err.message || "Failed to load images"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchImages(); }, []);

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
        <h1 className="page-title">Scan History</h1>
        {!loading && images.length > 0 && (
          <div className="search-bar">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search by filename&hellip;"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="search-input"
            />
          </div>
        )}
      </div>

      {error && (
        <div className="auth-error" style={{ marginBottom: "1rem" }}>
          <AlertCircle size={18} />
          <span>{error}</span>
          <button className="btn btn-ghost" onClick={fetchImages} style={{ marginLeft: "auto" }}>
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
              Upload Your First Scan
            </button>
          )}
        </div>
      )}

      {!loading && !error && filtered.length > 0 && (
        <div className="history-grid">
          {filtered.map((img) => (
            <div key={img.id} className="history-card">
              <div className="history-card-img"
                onClick={() => onViewImage(img.id, img.detections.length > 0 ? detectionsToPredictionResult(img.detections) : null)}>
                <img src={getImageUrl(img.id)} alt={img.original_name} loading="lazy" />
              </div>
              <div className="history-card-body">
                <p className="history-name" title={img.original_name}>{img.original_name}</p>
                <p className="history-meta">{new Date(img.uploaded_at).toLocaleDateString()}</p>
              </div>
              <div className="history-card-actions">
                <button className="btn btn-tiny" onClick={() => handleDownloadReport(img.id)}
                  title="Download Report">
                  <FileText size={14} />
                </button>
                <button className="btn btn-tiny btn-danger"
                  onClick={() => handleDelete(img.id)}
                  disabled={deletingId === img.id}
                  title="Delete">
                  {deletingId === img.id ? <span className="spinner-small" /> : <Trash2 size={14} />}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
