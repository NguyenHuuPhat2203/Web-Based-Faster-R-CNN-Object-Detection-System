import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  Image as ImageIcon,
  Trash2,
  FileText,
  Search,
  ArrowUpDown,
  Upload,
  RotateCcw,
} from "lucide-react";
import { useNavigate, useOutletContext } from "react-router-dom";
import {
  messageOf,
  listImages,
  deleteImage,
  downloadReport,
  fetchImageBlob,
  confirmedDetections,
  imageToPredictionResult,
  predict,
  type ImageInfo,
} from "../lib/api";
import {
  filterHistory,
  otherModel,
  type HistoryDetectionFilter,
  type HistoryModelFilter,
  type HistorySort,
} from "../lib/historyList";
import { useImageBlobUrls } from "../lib/useImageBlobUrls";
import type { LayoutOutletContext } from "./DashboardLayout";

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
  return null;
}

export function History() {
  const navigate = useNavigate();
  const { onViewImage } = useOutletContext<LayoutOutletContext>();
  const [images, setImages] = useState<ImageInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  // Row-level failures (delete, report, re-run) must not hide the loaded list.
  const [actionError, setActionError] = useState("");
  const [search, setSearch] = useState("");
  const [modelFilter, setModelFilter] = useState<HistoryModelFilter>("all");
  const [detectionFilter, setDetectionFilter] = useState<HistoryDetectionFilter>("all");
  const [sort, setSort] = useState<HistorySort>("newest");
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [rerunningId, setRerunningId] = useState<number | null>(null);

  const fetchImages = useCallback((showSpinner = true) => {
    if (showSpinner) setLoading(true);
    setError("");
    listImages()
      .then(setImages)
      .catch((err) => setError(messageOf(err, "Failed to load images")))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { fetchImages(); }, [fetchImages]);

  const thumbnails = useImageBlobUrls(images.map((img) => img.id));

  const handleDelete = async (id: number) => {
    if (!window.confirm("Delete this scan? This action cannot be undone.")) return;
    setDeletingId(id);
    try {
      await deleteImage(id);
      setImages((prev) => prev.filter((img) => img.id !== id));
    } catch (err) {
      setActionError(messageOf(err, "Failed to delete image"));
    } finally {
      setDeletingId(null);
    }
  };

  const handleDownloadReport = async (id: number) => {
    try {
      await downloadReport(id);
    } catch (err) {
      setActionError(messageOf(err, "Failed to generate report"));
    }
  };

  const handleRerun = async (img: ImageInfo) => {
    setRerunningId(img.id);
    setActionError("");
    try {
      const blob = await fetchImageBlob(img.id);
      const file = new File([blob], img.original_name, {
        type: img.mime_type || "image/png",
      });
      await predict(file, img.threshold ?? 0.5, otherModel(img.model_type));
      fetchImages(false);
    } catch (err) {
      setActionError(messageOf(err, "Re-run failed"));
    } finally {
      setRerunningId(null);
    }
  };

  const filtered = filterHistory(images, {
    search,
    model: modelFilter,
    detections: detectionFilter,
    sort,
  });
  const hasFilters =
    Boolean(search.trim()) || modelFilter !== "all" || detectionFilter !== "all";

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
            <select
              className="search-input"
              value={modelFilter}
              onChange={(e) => setModelFilter(e.target.value as HistoryModelFilter)}
              title="Filter by model"
              style={{ width: "auto", padding: "0.5rem 0.6rem" }}
            >
              <option value="all">All models</option>
              <option value="faster-rcnn">Faster R-CNN</option>
              <option value="yolov8">YOLOv8</option>
            </select>
            <select
              className="search-input"
              value={detectionFilter}
              onChange={(e) => setDetectionFilter(e.target.value as HistoryDetectionFilter)}
              title="Filter by detections"
              style={{ width: "auto", padding: "0.5rem 0.6rem" }}
            >
              <option value="all">All scans</option>
              <option value="with">With detections</option>
              <option value="without">No detections</option>
            </select>
            <button
              className="btn btn-ghost"
              style={{ width: "auto", padding: "0.5rem 0.75rem" }}
              title={sort === "newest" ? "Newest first" : "Oldest first"}
              onClick={() => setSort(sort === "newest" ? "oldest" : "newest")}
            >
              <ArrowUpDown size={16} />
              {sort === "newest" ? "Newest" : "Oldest"}
            </button>
          </div>
        )}
      </div>

      {(error || actionError) && (
        <div className="auth-error" style={{ marginBottom: "1rem" }}>
          <AlertCircle size={18} />
          <span>{error || actionError}</span>
          <button className="btn btn-ghost" onClick={error ? () => fetchImages() : () => setActionError("")} style={{ marginLeft: "auto", width: "auto", padding: "0.3rem 0.8rem" }}>
            {error ? "Retry" : "Dismiss"}
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
          <p>{hasFilters ? "No images match your filters." : "No scans yet. Upload an MRI to get started."}</p>
          {!hasFilters && (
            <button className="btn btn-primary" onClick={() => navigate("/detector")}>
              <Upload size={16} /> Upload Your First Scan
            </button>
          )}
        </div>
      )}

      {!loading && filtered.length > 0 && (
        <div className="history-list">
          {filtered.map((img) => {
            const scanBadge = getScanBadge(img.original_name);
            return (
              <div
                key={img.id}
                className="history-list-item"
                onClick={() => onViewImage(img.id, imageToPredictionResult(img))}
              >
                {thumbnails[img.id] ? (
                  <img
                    src={thumbnails[img.id]}
                    alt={img.original_name}
                    className="history-list-thumb"
                  />
                ) : (
                  <div className="history-list-thumb" />
                )}
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
                    <span>{confirmedDetections(img).length} detections</span>
                    {img.model_type && (
                      <>
                        <span style={{ color: "var(--text-secondary)" }}>•</span>
                        <span>
                          {img.model_type}
                          {img.model_version ? ` ${img.model_version}` : ""}
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <div className="history-list-actions">
                  <button
                    className="btn btn-tiny"
                    onClick={(e) => { e.stopPropagation(); handleRerun(img); }}
                    disabled={rerunningId === img.id}
                    title={`Re-run with ${otherModel(img.model_type)}`}
                  >
                    {rerunningId === img.id ? <span className="spinner-small" /> : <RotateCcw size={14} />}
                  </button>
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