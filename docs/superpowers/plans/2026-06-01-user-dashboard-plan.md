# User Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sidebar-based dashboard that becomes the authenticated app shell, replacing the current direct-to-detector flow.

**Architecture:** The authenticated part of the app is wrapped in a `DashboardLayout` with a persistent `Sidebar` and a content area that switches between Dashboard Home (stats), Detector (MRI analysis), History (image management), and Settings (profile). Three new backend endpoints support the new pages.

**Tech Stack:** FastAPI (backend), React + TypeScript + Vite (frontend), SQLAlchemy (ORM), lucide-react (icons)

---

## File Structure

### Backend (modified):
- `backend/routers/auth.py` — add `PUT /auth/profile`, `PUT /auth/password`, `GET /auth/stats`
- `backend/schemas.py` — add request/response schemas for new endpoints

### Frontend (new):
- `frontend/src/components/DashboardLayout.tsx` — sidebar + content area shell
- `frontend/src/components/Sidebar.tsx` — persistent navigation sidebar
- `frontend/src/components/DashboardHome.tsx` — stats overview page
- `frontend/src/components/Detector.tsx` — extracted from inline App.tsx
- `frontend/src/components/History.tsx` — dedicated image management page
- `frontend/src/components/Settings.tsx` — profile editing & password change

### Frontend (modified):
- `frontend/src/App.tsx` — rewire routing to use DashboardLayout
- `frontend/src/components/Navbar.tsx` — removed (replaced by Sidebar)
- `frontend/src/lib/api.ts` — add stats, profile update, password change API calls

---

### Task 1: Backend — Add schemas for new endpoints

**Files:**
- Modify: `backend/schemas.py`

- [ ] **Step 1: Add new schemas**

Add these to the existing `backend/schemas.py` file, inside the `# ── Auth ──` section:

```python
# ── Profile ──────────────────────────────────────────────

class ProfileUpdateRequest(BaseModel):
    username: str | None = None
    email: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str | None) -> str | None:
        if v is not None and "@" not in v:
            raise ValueError("Invalid email")
        return v


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


# ── Stats ────────────────────────────────────────────────

class UserStats(BaseModel):
    total_scans: int
    total_detections: int
    scans_this_month: int
```

- [ ] **Step 2: Verify file parses**

```bash
cd backend && python -c "from schemas import ProfileUpdateRequest, PasswordChangeRequest, UserStats; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/schemas.py
git commit -m "feat: add schemas for profile, password, and stats endpoints"
```

---

### Task 2: Backend — Add profile, password, and stats endpoints

**Files:**
- Modify: `backend/routers/auth.py`

- [ ] **Step 1: Add imports**

After the existing `from auth import` block in `backend/routers/auth.py`, add:

```python
from schemas import (
    RegisterRequest,
    LoginRequest,
    RefreshRequest,
    LogoutRequest,
    UserWithTokens,
    TokenResponse,
    ProfileUpdateRequest,     # new
    PasswordChangeRequest,    # new
    UserStats,                # new
)
```

- [ ] **Step 2: Add `PUT /auth/profile` endpoint**

Add before the `# ── Google OAuth ──` section:

```python
# ── Profile Update ───────────────────────────────────────

@router.put("/profile", response_model=UserOut)
def update_profile(
    body: ProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.username is not None:
        current_user.username = body.username
    if body.email is not None:
        existing = db.query(User).filter(
            User.email == body.email, User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already in use",
            )
        current_user.email = body.email
    db.commit()
    db.refresh(current_user)
    return current_user


# ── Password Change ──────────────────────────────────────

@router.put("/password")
def change_password(
    body: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not current_user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change password for Google-authenticated accounts",
        )
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )
    current_user.hashed_password = hash_password(body.new_password)
    db.commit()
    return {"message": "Password updated successfully"}


# ── User Stats ───────────────────────────────────────────

@router.get("/stats", response_model=UserStats)
def user_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    total_scans = db.query(Image).filter(
        Image.user_id == current_user.id, ~Image.deleted
    ).count()

    scans_this_month = db.query(Image).filter(
        Image.user_id == current_user.id,
        ~Image.deleted,
        Image.uploaded_at >= now.replace(day=1),
    ).count()

    total_detections = db.query(Detection).join(Image).filter(
        Image.user_id == current_user.id,
        ~Image.deleted,
    ).count()

    return UserStats(
        total_scans=total_scans,
        total_detections=total_detections,
        scans_this_month=scans_this_month,
    )
```

- [ ] **Step 3: Add Image/Detection import**

Add to the existing model imports at the top:
```python
from models import User, TokenBlacklist, Image, Detection
```
(Currently only `User` and `TokenBlacklist` are imported — replace the line.)

- [ ] **Step 4: Run existing tests to verify no regressions**

```bash
cd backend && python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/routers/auth.py
git commit -m "feat: add profile, password, and stats endpoints"
```

---

### Task 3: Frontend — Add new API client functions

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Add stats, profile, and password API functions**

Add after the `getReportUrl()` function at the bottom of `api.ts`:

```typescript
/* ---------- Stats API ---------- */

export interface UserStats {
  total_scans: number;
  total_detections: number;
  scans_this_month: number;
}

export async function fetchUserStats(): Promise<UserStats> {
  const res = await authFetch("/auth/stats");
  if (!res.ok) throw new Error("Failed to fetch stats");
  return res.json();
}

/* ---------- Profile API ---------- */

export interface ProfileUpdate {
  username?: string;
  email?: string;
}

export async function updateProfile(data: ProfileUpdate): Promise<UserInfo> {
  const res = await authFetch("/auth/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Update failed" }));
    throw new Error(err.detail || "Update failed");
  }
  return res.json();
}

/* ---------- Password API ---------- */

export async function changePassword(
  currentPassword: string,
  newPassword: string,
): Promise<void> {
  const res = await authFetch("/auth/password", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: currentPassword,
      new_password: newPassword,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Password change failed" }));
    throw new Error(err.detail || "Password change failed");
  }
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: No errors (or only unavoidable ones).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat: add API functions for stats, profile, and password"
```

---

### Task 4: Frontend — Create Sidebar component

**Files:**
- Create: `frontend/src/components/Sidebar.tsx`

- [ ] **Step 1: Write the Sidebar component**

```tsx
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
        {!collapsed && <span className="sidebar-brand">Brain Tumor<br/>Detector</span>}
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
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Sidebar.tsx
git commit -m "feat: add Sidebar component with navigation and collapse"
```

---

### Task 5: Frontend — Create DashboardLayout component

**Files:**
- Create: `frontend/src/components/DashboardLayout.tsx`

- [ ] **Step 1: Write the DashboardLayout component**

```tsx
import React, { useState } from "react";
import { Sidebar, type DashboardPage } from "./Sidebar";
import { DashboardHome } from "./DashboardHome";
import { Detector } from "./Detector";
import { History } from "./History";
import { Settings } from "./Settings";

export function DashboardLayout() {
  const [page, setPage] = useState<DashboardPage>("home");

  const renderPage = () => {
    switch (page) {
      case "home":
        return <DashboardHome onNavigate={setPage} />;
      case "detector":
        return <Detector />;
      case "history":
        return <History onNavigate={setPage} />;
      case "settings":
        return <Settings />;
    }
  };

  return (
    <div className="dashboard-layout">
      <Sidebar activePage={page} onNavigate={setPage} />
      <main className="dashboard-content">
        {renderPage()}
      </main>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: Errors for missing DashboardHome, Detector, History, Settings imports — those come next.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/DashboardLayout.tsx
git commit -m "feat: add DashboardLayout shell with sidebar routing"
```

---

### Task 6: Frontend — Create DashboardHome (stats overview)

**Files:**
- Create: `frontend/src/components/DashboardHome.tsx`

- [ ] **Step 1: Write the DashboardHome component**

```tsx
import React, { useEffect, useState } from "react";
import {
  Scan,
  AlertCircle,
  Brain,
  CalendarDays,
  ArrowRight,
  Image as ImageIcon,
} from "lucide-react";
import { fetchUserStats, listImages, getImageUrl, type UserStats, type ImageInfo } from "../lib/api";
import type { DashboardPage } from "./Sidebar";

interface DashboardHomeProps {
  onNavigate: (page: DashboardPage) => void;
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
        <div className="loading-state"><div className="spinner" /><p>Loading dashboard…</p></div>
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

      {/* ── Stat Cards ── */}
      <div className="stats-grid">
        <div className="stat-card">
          <Scan size={24} className="stat-icon" />
          <div>
            <p className="stat-label">Total Scans</p>
            <p className="stat-value">{stats?.total_scans ?? 0}</p>
          </div>
        </div>
        <div className="stat-card">
          <Brain size={24} className="stat-icon" />
          <div>
            <p className="stat-label">Detections Found</p>
            <p className="stat-value">{stats?.total_detections ?? 0}</p>
          </div>
        </div>
        <div className="stat-card">
          <CalendarDays size={24} className="stat-icon" />
          <div>
            <p className="stat-label">Scans This Month</p>
            <p className="stat-value">{stats?.scans_this_month ?? 0}</p>
          </div>
        </div>
      </div>

      {/* ── Quick Actions ── */}
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

      {/* ── Recent Activity ── */}
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
                onClick={() => onNavigate("detector")}
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
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: No errors (DashboardHome uses existing types from api.ts).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/DashboardHome.tsx
git commit -m "feat: add DashboardHome with stats, quick actions, and recent activity"
```

---

### Task 7: Frontend — Extract Detector component

**Files:**
- Create: `frontend/src/components/Detector.tsx`
- Modify: `frontend/src/App.tsx` (will be done in a later task)

- [ ] **Step 1: Create Detector.tsx**

This extracts the inline detector code from `App.tsx` (lines ~426-604) into its own component. The component receives no props (DashboardLayout handles navigation). The refs and state that were in App stay inside Detector.

Copy the detector-specific logic from App.tsx — everything inside the `return (...)` block after the `/* ---------- Main detector page ---------- */` comment (lines ~426-604) and the associated state/handlers (lines ~57-398) into a new standalone component.

The new `Detector.tsx` file:

```tsx
import React, { useState, useRef, useCallback, useEffect } from "react";
import {
  Upload, AlertCircle, Download, Sliders, Clock,
  Image as ImageIcon, RotateCcw, Camera, Cpu, FileText,
  Thermometer, Ruler, Eye,
} from "lucide-react";
import {
  predict,
  listImages,
  getImageUrl,
  getReportUrl,
  authFetch,
  type PredictionResult,
  type ImageInfo,
  type ModelType,
} from "../lib/api";

/* ---------- Label map (mirrors backend CLASS_NAMES) ---------- */

const LABEL_NAMES: Record<number, string> = {
  1: "Brain tumor",
  2: "Brain tumor",
};

/* ---------- Model options ---------- */

const MODELS: { value: ModelType; label: string }[] = [
  { value: "faster-rcnn", label: "Faster R-CNN" },
  { value: "yolov8", label: "YOLOv8" },
];

/* ---------- Types ---------- */

type ViewMode = "original" | "annotated" | "split";

interface Measurement {
  x1: number; y1: number; x2: number; y2: number;
}

export function Detector() {
  // -- Mode --
  const [useWebcam, setUseWebcam] = useState(false);

  // -- Detection state --
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loadingPred, setLoadingPred] = useState(false);
  const [results, setResults] = useState<PredictionResult | null>(null);
  const [detectError, setDetectError] = useState("");
  const [threshold, setThreshold] = useState(0.5);
  const [model, setModel] = useState<ModelType>("faster-rcnn");
  const [latestImageId, setLatestImageId] = useState<number | null>(null);

  // -- View mode --
  const [viewMode, setViewMode] = useState<ViewMode>("annotated");
  const [showHeatmap, setShowHeatmap] = useState(false);

  // -- Measurement tool --
  const [measuring, setMeasuring] = useState(false);
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [measureStart, setMeasureStart] = useState<{ x: number; y: number } | null>(null);

  // -- Drag-and-drop --
  const [isDragging, setIsDragging] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const heatmapImgRef = useRef<HTMLImageElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  /* ---------- File handling ---------- */

  const loadFile = useCallback((file: File) => {
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResults(null);
    setDetectError("");
    setLatestImageId(null);
    setMeasurements([]);
    setShowHeatmap(false);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); e.stopPropagation(); setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) loadFile(file);
  };

  /* ---------- Drawing helpers ---------- */

  const drawAnnotations = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    mode: "detections" | "measurements" | "all",
  ) => {
    if (results && (mode === "detections" || mode === "all")) {
      ctx.lineWidth = 4;
      ctx.font = "bold 22px Inter, system-ui, sans-serif";

      results.boxes.forEach((box, i) => {
        const [xmin, ymin, xmax, ymax] = box;
        const score = results.scores[i];
        const label = results.label_names?.[i] || LABEL_NAMES[results.labels[i]] || `Class ${results.labels[i]}`;

        ctx.strokeStyle = "#10b981";
        ctx.strokeRect(xmin, ymin, xmax - xmin, ymax - ymin);

        const labelText = `${label}: ${(score * 100).toFixed(1)}%`;
        const textWidth = ctx.measureText(labelText).width;

        ctx.fillStyle = "#10b981";
        ctx.fillRect(xmin, ymin - 34, textWidth + 14, 34);
        ctx.fillStyle = "white";
        ctx.fillText(labelText, xmin + 7, ymin - 9);
      });
    }

    if (mode === "measurements" || mode === "all") {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.font = "bold 14px Inter, system-ui, sans-serif";

      for (const m of measurements) {
        ctx.beginPath();
        ctx.moveTo(m.x1, m.y1);
        ctx.lineTo(m.x2, m.y2);
        ctx.stroke();

        ctx.setLineDash([]);
        ctx.fillStyle = "#f59e0b";
        ctx.beginPath();
        ctx.arc(m.x1, m.y1, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(m.x2, m.y2, 5, 0, Math.PI * 2);
        ctx.fill();

        const dist = Math.sqrt((m.x2 - m.x1) ** 2 + (m.y2 - m.y1) ** 2);
        const mx = (m.x1 + m.x2) / 2;
        const my = (m.y1 + m.y2) / 2;
        const label = `${dist.toFixed(0)}px`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(245, 158, 11, 0.9)";
        ctx.fillRect(mx - tw / 2 - 4, my - 12, tw + 8, 20);
        ctx.fillStyle = "white";
        ctx.fillText(label, mx - tw / 2, my + 4);

        ctx.setLineDash([6, 4]);
      }
      ctx.setLineDash([]);

      if (measureStart) {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
      }
      ctx.setLineDash([]);
    }
  }, [results, measurements, measureStart]);

  const renderCanvas = useCallback(() => {
    if (!canvasRef.current || !imgRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;
    if (!ctx) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    if (viewMode === "original") {
      ctx.drawImage(img, 0, 0);
      return;
    }

    ctx.drawImage(img, 0, 0);

    if (viewMode === "annotated" || viewMode === "split") {
      drawAnnotations(ctx, canvas.width, canvas.height, "all");
    }
  }, [viewMode, drawAnnotations]);

  useEffect(() => {
    if (results) renderCanvas();
  }, [results, renderCanvas, showHeatmap, viewMode, measurements, measureStart]);

  useEffect(() => {
    if (!showHeatmap || !results?.heatmap || !canvasRef.current || !imgRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = imgRef.current;
    if (!ctx) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    if (viewMode === "original") {
      ctx.drawImage(img, 0, 0);
      return;
    }

    ctx.drawImage(img, 0, 0);

    const hmImg = new window.Image();
    hmImg.onload = () => {
      ctx.globalAlpha = 0.5;
      ctx.drawImage(hmImg, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1.0;
      drawAnnotations(ctx, canvas.width, canvas.height, "all");
    };
    hmImg.src = `data:image/png;base64,${results.heatmap}`;
  }, [showHeatmap, results, viewMode, drawAnnotations]);

  /* ---------- Predict ---------- */

  const handleDetect = async () => {
    if (!image) return;
    setDetectError("");
    setLoadingPred(true);
    setLatestImageId(null);
    try {
      const data = await predict(image, threshold, model);
      setResults(data);
    } catch (err: any) {
      setDetectError(err.message || "Detection failed");
    } finally {
      setLoadingPred(false);
    }
  };

  /* ---------- Download ---------- */

  const downloadAnnotated = () => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const link = document.createElement("a");
    link.download = `annotated-${image.name.replace(/\.[^.]+$/, "")}.png`;
    canvas.toBlob((blob) => {
      if (blob) { link.href = URL.createObjectURL(blob); link.click(); }
    });
  };

  /* ---------- Report ---------- */

  const downloadReport = async () => {
    if (!latestImageId) return;
    const url = getReportUrl(latestImageId);
    const resp = await authFetch(url);
    if (!resp.ok) { setDetectError("Failed to generate report"); return; }
    const blob = await resp.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `report-${latestImageId}.pdf`;
    link.click();
  };

  /* ---------- Measurement tool ---------- */

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!measuring || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    if (!measureStart) {
      setMeasureStart({ x, y });
    } else {
      setMeasurements(prev => [...prev, { x1: measureStart.x, y1: measureStart.y, x2: x, y2: y }]);
      setMeasureStart(null);
    }
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!measuring || !measureStart || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx || !imgRef.current) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    canvas.width = imgRef.current.naturalWidth;
    canvas.height = imgRef.current.naturalHeight;
    ctx.drawImage(imgRef.current, 0, 0);

    drawAnnotations(ctx, canvas.width, canvas.height, "all");

    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(measureStart.x, measureStart.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  /* ---------- Reset ---------- */

  const reset = () => {
    setImage(null);
    setPreview(null);
    setResults(null);
    setDetectError("");
    setLatestImageId(null);
    setMeasurements([]);
    setMeasureStart(null);
    setShowHeatmap(false);
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Brain Tumor Detector</h1>

      {detectError && (
        <div className="auth-error" style={{ marginBottom: "1rem" }}>
          <AlertCircle size={18} />
          <span>{detectError}</span>
        </div>
      )}

      {/* ── Toolbar ── */}
      <div className="toolbar">
        <div className="toolbar-group">
          <Cpu size={16} />
          <select className="model-select" value={model}
            onChange={(e) => setModel(e.target.value as ModelType)}>
            {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
        </div>
        <button className={`btn btn-ghost ${useWebcam ? "active" : ""}`}
          onClick={() => { setUseWebcam(!useWebcam); reset(); }}>
          <Camera size={16} /> {useWebcam ? "Upload Instead" : "Live Webcam"}
        </button>
      </div>

      <div className="glass-card">
        {useWebcam ? (
          <WebcamCapture model={model} threshold={threshold}
            onResult={(r) => setResults(r)} onError={(e) => setDetectError(e)} />
        ) : !preview ? (
          <label className={`upload-zone${isDragging ? " dragover" : ""}`}
            onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>
            <input ref={fileInputRef} type="file" hidden accept="image/*" onChange={handleFileChange} />
            <Upload size={48} className="mb-4" />
            <p>Click or drag image to upload</p>
            <p className="text-muted">Supports JPG, PNG &mdash; MRI scans preferred</p>
          </label>
        ) : (
          <div className="text-center">
            <div className="preview-container">
              <img ref={imgRef} src={preview} alt="Preview"
                style={{ display: results && viewMode !== "original" ? "none" : "block", maxWidth: "100%" }}
                onLoad={() => results && renderCanvas()} />
              <canvas ref={canvasRef}
                style={{ display: results && viewMode !== "original" ? "block" : "none", cursor: measuring ? "crosshair" : "default" }}
                onClick={handleCanvasClick} onMouseMove={handleCanvasMove} />
              {results?.heatmap && (
                <img ref={heatmapImgRef} src={`data:image/png;base64,${results.heatmap}`}
                  style={{ display: "none" }} alt="" />
              )}
            </div>

            {/* ── Threshold slider ── */}
            <div className="threshold-row">
              <Sliders size={16} />
              <label className="threshold-label">
                Confidence: <strong>{(threshold * 100).toFixed(0)}%</strong>
              </label>
              <input type="range" min="0.05" max="0.95" step="0.05" value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))} className="threshold-slider" />
            </div>

            {/* ── Action buttons ── */}
            <div className="action-row">
              <button className="btn btn-secondary" onClick={reset}><RotateCcw size={18} /> Reset</button>
              <button className="btn btn-primary" onClick={handleDetect} disabled={loadingPred}>
                {loadingPred ? <><span className="spinner-small" /> Processing&hellip;</> : "Detect"}
              </button>
              {results && <button className="btn btn-accent" onClick={downloadAnnotated}><Download size={18} /> Download</button>}
              {results && <button className="btn btn-secondary" onClick={downloadReport}><FileText size={18} /> Report</button>}
            </div>

            {/* ── View mode + extras toolbar ── */}
            {results && (
              <div className="view-toolbar">
                <div className="toolbar-group">
                  <Eye size={14} />
                  {(["annotated", "original", "split"] as ViewMode[]).map(mode => (
                    <button key={mode}
                      className={`btn btn-tiny ${viewMode === mode ? "active" : ""}`}
                      onClick={() => setViewMode(mode)}>
                      {mode === "annotated" ? "Annotated" : mode === "original" ? "Original" : "Side-by-side"}
                    </button>
                  ))}
                </div>

                {results.heatmap && (
                  <button className={`btn btn-tiny ${showHeatmap ? "active" : ""}`}
                    onClick={() => setShowHeatmap(!showHeatmap)}>
                    <Thermometer size={14} /> {showHeatmap ? "Hide Heatmap" : "Heatmap"}
                  </button>
                )}

                <button className={`btn btn-tiny ${measuring ? "active" : ""}`}
                  onClick={() => { setMeasuring(!measuring); if (!measuring) setMeasureStart(null); }}>
                  <Ruler size={14} /> {measuring ? "Done Measuring" : "Measure"}
                </button>

                {measurements.length > 0 && (
                  <button className="btn btn-tiny" onClick={() => { setMeasurements([]); setMeasureStart(null); }}>
                    Clear Lines
                  </button>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Results section ── */}
      {results && !useWebcam && results.boxes.length > 0 && (
        <div className="stats">
          <div className="stat-card"><h3>Detections</h3><p className="stat-value">{results.boxes.length}</p></div>
          <div className="stat-card"><h3>Avg Score</h3><p className="stat-value">
            {((results.scores.reduce((a, b) => a + b, 0) / results.scores.length) * 100).toFixed(1)}%
          </p></div>
          {results.label_names?.length > 0 && (
            <div className="stat-card"><h3>Top Class</h3><p className="stat-value-sm">{results.label_names[0]}</p></div>
          )}
        </div>
      )}

      {results && !useWebcam && results.boxes.length === 0 && (
        <div className="no-detections">
          <AlertCircle size={24} />
          <p><strong>No tumors detected</strong> above {(threshold * 100).toFixed(0)}% confidence.</p>
          <p className="text-muted">Try lowering the threshold or uploading a different scan.</p>
        </div>
      )}
    </div>
  );
}
```

Note: This component needs the `WebcamCapture` import at the top. Let me add it:

```tsx
import { WebcamCapture } from "./WebcamCapture";
```

Add this alongside the other component imports.

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Detector.tsx
git commit -m "feat: extract Detector from inline App.tsx into standalone component"
```

---

### Task 8: Frontend — Create History page

**Files:**
- Create: `frontend/src/components/History.tsx`

- [ ] **Step 1: Write the History component**

```tsx
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
  type ImageInfo,
} from "../lib/api";
import type { DashboardPage } from "./Sidebar";

interface HistoryProps {
  onNavigate: (page: DashboardPage) => void;
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
              placeholder="Search by filename…"
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
          <p>Loading history…</p>
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
              <div className="history-card-img" onClick={() => onNavigate("detector")}>
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
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/History.tsx
git commit -m "feat: add History page with search, delete, and report download"
```

---

### Task 9: Frontend — Create Settings page

**Files:**
- Create: `frontend/src/components/Settings.tsx`

- [ ] **Step 1: Write the Settings component**

```tsx
import React, { useState } from "react";
import {
  User,
  Mail,
  Lock,
  AlertCircle,
  CheckCircle,
  Eye,
  EyeOff,
  CalendarDays,
  Scan,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { updateProfile, changePassword } from "../lib/api";

export function Settings() {
  const { isAuth } = useAuth();

  // Profile state
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [profileMsg, setProfileMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [profileSubmitting, setProfileSubmitting] = useState(false);

  // Password state
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPasswords, setShowPasswords] = useState(false);
  const [passwordMsg, setPasswordMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);
  const [passwordSubmitting, setPasswordSubmitting] = useState(false);

  const handleProfileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setProfileMsg(null);
    setProfileSubmitting(true);
    try {
      const data: Record<string, string> = {};
      if (username.trim()) data.username = username.trim();
      if (email.trim()) data.email = email.trim();
      if (Object.keys(data).length === 0) {
        setProfileMsg({ type: "error", text: "No changes to save." });
        setProfileSubmitting(false);
        return;
      }
      await updateProfile(data);
      setProfileMsg({ type: "success", text: "Profile updated successfully!" });
      setUsername("");
      setEmail("");
    } catch (err: any) {
      setProfileMsg({ type: "error", text: err.message || "Update failed" });
    } finally {
      setProfileSubmitting(false);
    }
  };

  const handlePasswordSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setPasswordMsg(null);

    if (newPassword !== confirmPassword) {
      setPasswordMsg({ type: "error", text: "Passwords do not match" });
      return;
    }
    if (newPassword.length < 6) {
      setPasswordMsg({ type: "error", text: "Password must be at least 6 characters" });
      return;
    }

    setPasswordSubmitting(true);
    try {
      await changePassword(currentPassword, newPassword);
      setPasswordMsg({ type: "success", text: "Password changed successfully!" });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err: any) {
      setPasswordMsg({ type: "error", text: err.message || "Password change failed" });
    } finally {
      setPasswordSubmitting(false);
    }
  };

  return (
    <div className="page-container">
      <h1 className="page-title">Settings</h1>

      {/* ── Profile ── */}
      <div className="settings-card">
        <h2 className="settings-section-title"><User size={20} /> Profile</h2>
        {profileMsg && (
          <div className={`auth-${profileMsg.type === "error" ? "error" : "success"}`}>
            {profileMsg.type === "error" ? <AlertCircle size={18} /> : <CheckCircle size={18} />}
            <span>{profileMsg.text}</span>
          </div>
        )}
        <form onSubmit={handleProfileSubmit}>
          <div className="input-group">
            <User size={20} />
            <input
              type="text"
              placeholder="New username (leave blank to keep current)"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              minLength={2}
            />
          </div>
          <div className="input-group">
            <Mail size={20} />
            <input
              type="email"
              placeholder="New email (leave blank to keep current)"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <button className="btn btn-primary" type="submit" disabled={profileSubmitting}>
            {profileSubmitting ? "Saving…" : "Save Changes"}
          </button>
        </form>
      </div>

      {/* ── Password ── */}
      <div className="settings-card">
        <h2 className="settings-section-title"><Lock size={20} /> Change Password</h2>
        {passwordMsg && (
          <div className={`auth-${passwordMsg.type === "error" ? "error" : "success"}`}>
            {passwordMsg.type === "error" ? <AlertCircle size={18} /> : <CheckCircle size={18} />}
            <span>{passwordMsg.text}</span>
          </div>
        )}
        <form onSubmit={handlePasswordSubmit}>
          <div className="input-group">
            <Lock size={20} />
            <input
              type={showPasswords ? "text" : "password"}
              placeholder="Current password"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <Lock size={20} />
            <input
              type={showPasswords ? "text" : "password"}
              placeholder="New password (min 6 chars)"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              minLength={6}
            />
          </div>
          <div className="input-group">
            <Lock size={20} />
            <input
              type={showPasswords ? "text" : "password"}
              placeholder="Confirm new password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
            />
          </div>
          <div className="settings-actions">
            <button className="btn btn-primary" type="submit" disabled={passwordSubmitting}>
              {passwordSubmitting ? "Updating…" : "Update Password"}
            </button>
            <button type="button" className="btn btn-ghost"
              onClick={() => setShowPasswords(!showPasswords)}>
              {showPasswords ? <EyeOff size={18} /> : <Eye size={18} />}
              {showPasswords ? "Hide" : "Show"} Passwords
            </button>
          </div>
        </form>
      </div>

      {/* ── Account Info ── */}
      <div className="settings-card">
        <h2 className="settings-section-title"><User size={20} /> Account Info</h2>
        <p className="text-muted">View-only account details.</p>
        {/* Account info would require a profile fetch endpoint — deferred */}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -20
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Settings.tsx
git commit -m "feat: add Settings page with profile editing and password change"
```

---

### Task 10: Frontend — Rewire App.tsx with DashboardLayout

**Files:**
- Modify: `frontend/src/App.tsx`
- Remove: `frontend/src/components/Navbar.tsx`

- [ ] **Step 1: Rewire App.tsx to use DashboardLayout**

Replace the current `App.tsx` content with a slimmed-down version that uses `DashboardLayout` instead of inline detector code:

```tsx
import { useState } from "react";
import { useAuth } from "./context/AuthContext";
import { Login } from "./components/Login";
import { Register } from "./components/Register";
import { OAuthCallback } from "./components/OAuthCallback";
import { DashboardLayout } from "./components/DashboardLayout";

type Page = "login" | "register" | "oauth-callback";

function App() {
  const { isAuth, loading } = useAuth();
  const [page, setPage] = useState<Page>(() => {
    const path = window.location.pathname;
    if (path.startsWith("/oauth-callback")) return "oauth-callback";
    return "login";
  });

  // Once authenticated, DashboardLayout handles all navigation
  useEffect(() => {
    if (isAuth && page !== "oauth-callback") {
      setPage("login"); // auth context handles redirect via DashboardLayout render
    }
  }, [isAuth, page]);

  if (loading) {
    return <div className="loading-screen"><div className="spinner" /></div>;
  }

  if (page === "oauth-callback") {
    return <OAuthCallback />;
  }

  if (!isAuth) {
    if (page === "register") {
      return <Register onSwitchToLogin={() => setPage("login")} />;
    }
    return <Login onSwitchToRegister={() => setPage("register")} />;
  }

  return <DashboardLayout />;
}

export default App;
```

Add the missing `useEffect` import to the existing React import:
```tsx
import { useState, useEffect } from "react";
```

This replaces the entire current `App.tsx` file content (all the detector-specific code is now in `Detector.tsx`).

- [ ] **Step 2: Remove Navbar (replaced by Sidebar)**

Delete `frontend/src/components/Navbar.tsx`:
```bash
rm frontend/src/components/Navbar.tsx
```

- [ ] **Step 3: Verify TypeScript compiles**

```bash
cd frontend && npx tsc --noEmit 2>&1 | head -30
```

Expected: No errors.

- [ ] **Step 4: Verify build works**

```bash
cd frontend && npx vite build 2>&1 | tail -10
```

Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx
git add -A  # captures Navbar deletion
git commit -m "feat: wire DashboardLayout into App, remove Navbar"
```

---

### Task 11: Frontend — Add CSS for sidebar, dashboard, history, settings

**Files:**
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Add sidebar and dashboard layout styles**

Add to `frontend/src/index.css`:

```css
/* ── Dashboard Layout ── */

.dashboard-layout {
  display: flex;
  min-height: 100vh;
}

.dashboard-content {
  flex: 1;
  margin-left: var(--sidebar-width);
  padding: 2rem;
  overflow-y: auto;
  transition: margin-left 0.2s ease;
}

/* ── Sidebar ── */

.sidebar {
  position: fixed;
  top: 0;
  left: 0;
  width: var(--sidebar-width);
  height: 100vh;
  background: var(--bg-card);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  z-index: 100;
  transition: width 0.2s ease;
  overflow: hidden;
}

.sidebar.collapsed {
  width: 60px;
}

.sidebar-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 1rem;
  border-bottom: 1px solid var(--border-color);
  color: var(--accent);
}

.sidebar-brand {
  font-weight: 700;
  font-size: 0.9rem;
  line-height: 1.2;
}

.sidebar-collapse-btn {
  margin-left: auto;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 4px;
}

.sidebar-collapse-btn:hover {
  color: var(--text);
  background: var(--bg-hover);
}

.sidebar-nav {
  flex: 1;
  padding: 0.5rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.sidebar-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  border-radius: 8px;
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 0.9rem;
  width: 100%;
  text-align: left;
  transition: all 0.15s ease;
}

.sidebar-item:hover {
  background: var(--bg-hover);
  color: var(--text);
}

.sidebar-item.active {
  background: var(--accent);
  color: white;
}

.sidebar-item.logout {
  margin-top: auto;
}

.sidebar-footer {
  padding: 0.5rem;
  border-top: 1px solid var(--border-color);
}

/* ── Page Containers ── */

.page-container {
  max-width: 960px;
  margin: 0 auto;
  width: 100%;
}

.page-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 1.5rem;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

/* ── Stats Grid ── */

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.stat-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 1.25rem;
  display: flex;
  align-items: center;
  gap: 1rem;
}

.stat-icon {
  color: var(--accent);
  opacity: 0.8;
}

.stat-label {
  color: var(--text-muted);
  font-size: 0.85rem;
  margin-bottom: 0.25rem;
}

.stat-value {
  font-size: 1.75rem;
  font-weight: 700;
  color: var(--text);
}

/* ── Sections ── */

.section {
  margin-bottom: 2rem;
}

.section-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 1rem;
}

/* ── Quick Actions ── */

.quick-actions {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.action-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
  transition: all 0.15s ease;
  color: var(--text-muted);
}

.action-card:hover {
  border-color: var(--accent);
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 8%, transparent);
}

/* ── Recent Activity ── */

.recent-list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.recent-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  border-radius: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  cursor: pointer;
  width: 100%;
  text-align: left;
  transition: background 0.15s ease;
}

.recent-item:hover {
  background: var(--bg-hover);
}

.recent-thumb {
  width: 48px;
  height: 48px;
  border-radius: 6px;
  object-fit: cover;
}

.recent-info {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
}

.recent-name {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--text);
}

.recent-date {
  font-size: 0.8rem;
  color: var(--text-muted);
}

/* ── Loading & Empty States ── */

.loading-state,
.error-state,
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 3rem;
  color: var(--text-muted);
  gap: 0.75rem;
  text-align: center;
}

.error-state .btn,
.empty-state .btn {
  margin-top: 0.5rem;
}

/* ── Search Bar ── */

.search-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 0.5rem 0.75rem;
  color: var(--text-muted);
}

.search-input {
  background: none;
  border: none;
  outline: none;
  color: var(--text);
  font-size: 0.9rem;
  min-width: 200px;
}

/* ── History Grid ── */

.history-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 1rem;
}

.history-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  overflow: hidden;
  transition: border-color 0.15s ease;
}

.history-card:hover {
  border-color: var(--accent);
}

.history-card-img {
  aspect-ratio: 4 / 3;
  overflow: hidden;
  cursor: pointer;
}

.history-card-img img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.2s ease;
}

.history-card-img img:hover {
  transform: scale(1.05);
}

.history-card-body {
  padding: 0.75rem;
}

.history-name {
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-meta {
  font-size: 0.78rem;
  color: var(--text-muted);
  margin-top: 0.2rem;
}

.history-card-actions {
  display: flex;
  gap: 0.25rem;
  padding: 0.5rem 0.75rem;
  border-top: 1px solid var(--border-color);
}

.history-card-actions .btn-tiny {
  padding: 0.3rem 0.5rem;
  font-size: 0.78rem;
}

/* ── Settings ── */

.settings-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
}

.settings-section-title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.settings-actions {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-top: 0.5rem;
}

.auth-success {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem;
  border-radius: 8px;
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
  margin-bottom: 1rem;
  font-size: 0.9rem;
}

/* ── CSS Variables (add to :root if not defined) ── */

:root {
  --sidebar-width: 220px;
}
```

- [ ] **Step 2: Verify by building the frontend**

```bash
cd frontend && npx vite build 2>&1 | tail -10
```

Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/index.css
git commit -m "feat: add CSS for sidebar, dashboard, history, and settings pages"
```

---

### Task 12: Verify everything works end-to-end

- [ ] **Step 1: Start the backend**

```bash
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 &
```

- [ ] **Step 2: Start the frontend dev server**

```bash
cd frontend && npx vite --port 8001 --host &
```

- [ ] **Step 3: Test the flow**
  1. Open http://localhost:8001 — should see Login page
  2. Register a new account — should redirect to Dashboard
  3. Verify sidebar navigation works (Dashboard Home, Detector, History, Settings)
  4. Test detector functionality (upload image, run detection)
  5. Test History (view, delete images)
  6. Test Settings (update profile, change password)

- [ ] **Step 4: Clean up background processes**

```bash
kill %1 %2 2>/dev/null; true
```

- [ ] **Step 5: Run `gitnexus_detect_changes()`** (per CLAUDE.md requirements)

```bash
npx gitnexus detect-changes 2>/dev/null || echo "Skipping GitNexus check"
```

- [ ] **Step 6: Final commit of any remaining changes**

```bash
git add -A && git status
```
