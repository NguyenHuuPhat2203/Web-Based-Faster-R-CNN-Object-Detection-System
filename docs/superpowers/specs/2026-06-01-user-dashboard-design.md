# User Dashboard — Design Spec

**Date:** 2026-06-01
**Project:** Brain Tumor Faster R-CNN Detection System

## Overview

Add a dashboard page that users land on after login/registration, providing a central hub for managing their data, accessing detection tools, and viewing account information.

## Architecture

### Component Tree

```
App (routing gate)
├── Login / Register / OAuthCallback  (unauthenticated)
└── DashboardLayout                   (authenticated – new)
    ├── Sidebar                        (new – persistent navigation)
    └── ContentArea
        ├── DashboardHome              (new – stats overview)
        ├── Detector                   (existing, moved into layout)
        ├── History                    (new – dedicated image management)
        └── Settings                   (new – profile & password)
```

### Routing

Simple state-based routing (no React Router needed, matching current pattern):

```
page: "login" | "register" | "oauth-callback"
dashboardPage: "home" | "detector" | "history" | "settings"
```

After successful auth, `App` renders `DashboardLayout` instead of inline detector code. `DashboardLayout` manages the `dashboardPage` state and renders the corresponding view in the content area.

The existing `Navbar` is replaced by the `Sidebar`.

## Navigation — Sidebar

- Persistent vertical sidebar on the left, dark-themed to match existing design
- Brand/logo at top ("Brain Tumor Detector" + Brain icon)
- Nav items with icons (from `lucide-react`): Dashboard (home), Detector (scan), History (clock), Settings (settings)
- Active item highlighted with accent color
- Bottom section: current user display (avatar/initials + email) + Sign Out button
- Collapsible for mobile (hamburger toggle)

## Page Views

### DashboardHome (stats overview)

- **Stat cards row:** Total Scans, Detections Found, Studies This Month (counts from backend)
- **Recent Activity:** last 5 uploaded images — thumbnail, filename, date, detection count summary; click opens Detector with that image loaded
- **Quick Actions:** "Upload New Scan" card → navigates to Detector

### Detector (existing, moved)

- Same functionality as today: upload image, select model (Faster R-CNN / YOLOv8), run detection, view annotations/heatmap, measure, download, generate PDF report
- The current "View History" toggle is removed from this page (moved to dedicated History page)
- Canvas rendering, measurement tool, all existing features preserved as-is

### History (new dedicated page)

- **Grid of cards** — thumbnails + filename + upload date + detection count
- **Delete** — per-image delete button with confirmation modal (soft delete, maps to existing backend endpoint)
- **Download report** — per-image PDF report download
- **Click image** — navigates to Detector and loads that image
- **Empty state:** "No scans yet — upload your first MRI" with CTA → Detector
- **Error state:** error message + retry button if loading fails

### Settings (new page)

- **Profile section:** editable username and email (with validation)
- **Change password:** current password + new password + confirm password (min 6 chars)
- **Account info:** read-only display: account creation date, total scans
- **Error/loading states** on all form submissions

## Backend Changes

### New endpoints (in `backend/routers/auth.py`):

| Method | Path | Description |
|--------|------|-------------|
| `PUT` | `/auth/profile` | Update username and/or email (auth required) |
| `PUT` | `/auth/password` | Change password (requires current password, auth required) |

### New endpoint (in backend or new router):

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/auth/stats` | Return user stats: total_scans, total_detections, scans_this_month (auth required) |

### Existing endpoints reused:
- `POST /auth/login`, `POST /auth/register` — no changes
- `POST /auth/logout` — no changes
- `GET /images` — used by History page (already user-scoped)
- `DELETE /images/{id}` — used by History page (already user-scoped)
- `GET /images/{id}` — used to load image and thumbnail
- `GET /images/{id}/report` — used to download PDF report

## Data Flow

1. User authenticates (login/register/OAuth) → tokens stored in localStorage
2. `AuthContext` updates `isAuth = true` → `App` renders `DashboardLayout`
3. `DashboardLayout` calls `GET /auth/stats` for home page, `GET /images` for history
4. All authenticated requests use the existing `authFetch()` wrapper
5. Detector page remains self-contained with its own state (image, results, measurements)

## Error Handling

- All data-fetching views (`DashboardHome`, `History`) implement: loading spinner, error message + retry, empty state
- All form views (`Settings`) implement: inline validation errors, server error display, success feedback
- The existing error patterns (`.auth-error` with `AlertCircle` icon) are reused consistently

## Out of Scope

- Account deletion ("Danger zone") — deferred
- Pagination/infinite scroll for large image libraries — deferred (future enhancement)
- Role-based access or admin panel
- Theme customization or dark/light toggle

## File Changes Summary

**New files (frontend):**
- `frontend/src/components/DashboardLayout.tsx` — sidebar + content area shell
- `frontend/src/components/Sidebar.tsx` — persistent navigation sidebar
- `frontend/src/components/DashboardHome.tsx` — stats overview page
- `frontend/src/components/Detector.tsx` — extracted from inline App.tsx (current analysis tool)
- `frontend/src/components/History.tsx` — dedicated image management page
- `frontend/src/components/Settings.tsx` — profile editing & password change

**Modified files (frontend):**
- `frontend/src/App.tsx` — replace inline detector rendering with `DashboardLayout`; remove history toggle from detector
- `frontend/src/components/Navbar.tsx` — replaced by Sidebar (Navbar removed)
- `frontend/src/lib/api.ts` — add new API functions for stats, profile update, password change

**Modified files (backend):**
- `backend/routers/auth.py` — add `PUT /auth/profile`, `PUT /auth/password`, `GET /auth/stats` endpoints
- `backend/schemas.py` — add request/response schemas for new endpoints
- `backend/auth.py` — add `verify_current_password` helper (optional)
