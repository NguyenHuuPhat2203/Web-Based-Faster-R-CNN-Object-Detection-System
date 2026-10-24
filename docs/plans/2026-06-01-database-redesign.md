# Database Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Normalize detection results, add image metadata/model provenance, introduce Alembic migrations, and add token blacklist cleanup.

**Architecture:** Keep SQLite + SQLAlchemy. Add `detections` table, extend `images` table with metadata columns, add Alembic for migration management, add utility for expired token cleanup.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.x, Alembic, SQLite

---

### Task 1: Install Alembic & Initialize

**Files:**
- Modify: `backend/requirements.txt` (or `pyproject.toml`)
- Create: `backend/alembic.ini`
- Create: `backend/alembic/`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Create: `backend/alembic/versions/`

- [ ] **Step 1: Install Alembic**

Run: `cd /d/Side_Project/Brain_Tumor_FasterRCNN/backend && pip install alembic`

- [ ] **Step 2: Initialize Alembic**

Run: `cd /d/Side_Project/Brain_Tumor_FasterRCNN/backend && alembic init alembic`

This creates `alembic.ini` and `alembic/` directory with env.py and script.py.mako.

- [ ] **Step 3: Configure Alembic to use DATABASE_URL from env**

Edit `backend/alembic.ini` — replace `sqlalchemy.url = driver://user:pass@localhost/dbname` with:
```
# Leave empty — read from env
sqlalchemy.url =
```

Edit `backend/alembic/env.py` — before the `target_metadata = None` line, add:
```python
import os
import sys
from dotenv import load_dotenv

# Ensure backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from database import Base
target_metadata = Base.metadata

# Read DATABASE_URL from env
config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL", "sqlite:///./brain_tumor.db"))
```

- [ ] **Step 4: Commit**

```bash
git add backend/alembic.ini backend/alembic/
git commit -m "chore: add Alembic database migrations"
```

---

### Task 2: Update Models (`backend/models.py`)

**Files:**
- Modify: `backend/models.py`

- [ ] **Step 1: Rewrite models.py**

Replace the entire file with:

```python
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Float
from sqlalchemy.orm import Session, relationship
from database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), nullable=False)
    hashed_password = Column(String(255), nullable=True)
    google_id = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    images = relationship("Image", back_populates="user", cascade="all, delete-orphan")


class TokenBlacklist(Base):
    __tablename__ = "token_blacklist"

    id = Column(Integer, primary_key=True, index=True)
    jti = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=_utcnow)


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    original_name = Column(String(500), nullable=False)
    stored_name = Column(String(255), nullable=False)
    filepath = Column(String(1000), nullable=False)
    mime_type = Column(String(50), nullable=False)
    uploaded_at = Column(DateTime, default=_utcnow)
    deleted = Column(Boolean, default=False)

    # Image metadata
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    content_hash = Column(String(64), nullable=True)

    # Model provenance
    model_type = Column(String(50), nullable=True)
    model_version = Column(String(100), nullable=True)
    threshold = Column(Float, nullable=True)

    user = relationship("User", back_populates="images")
    detections = relationship("Detection", back_populates="image",
                              cascade="all, delete-orphan",
                              passive_deletes=True)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)

    label = Column(Integer, nullable=False)
    label_name = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)

    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    image = relationship("Image", back_populates="detections")


def purge_expired_tokens(db: Session) -> None:
    """Remove expired tokens from the blacklist."""
    count = db.query(TokenBlacklist).filter(
        TokenBlacklist.expires_at < datetime.now(timezone.utc)
    ).delete()
    if count:
        db.commit()
```

- [ ] **Step 2: Verify models import correctly**

Run: `cd /d/Side_Project/Brain_Tumor_FasterRCNN/backend && python -c "from models import User, TokenBlacklist, Image, Detection, purge_expired_tokens; print('OK')"`
Expected: `OK`

---

### Task 3: Update Pydantic Schemas (`backend/schemas.py`)

**Files:**
- Modify: `backend/schemas.py`

- [ ] **Step 1: Rewrite schemas.py**

Replace file with:

```python
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, field_validator


# ── Auth ──────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    access_token: str
    refresh_token: str


# ── User ──────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: int
    email: str
    username: str
    created_at: datetime

    model_config = {"from_attributes": True}


class UserWithTokens(UserOut):
    access_token: str
    refresh_token: str


# ── Detection ─────────────────────────────────────────────────────────

class DetectionOut(BaseModel):
    label: int
    label_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    model_config = {"from_attributes": True}


# ── Image ─────────────────────────────────────────────────────────────

class ImageOut(BaseModel):
    id: int
    original_name: str
    mime_type: str
    uploaded_at: datetime
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    model_type: Optional[str] = None
    model_version: Optional[str] = None
    threshold: Optional[float] = None
    detections: list[DetectionOut] = []

    model_config = {"from_attributes": True}


class ImageList(BaseModel):
    images: list[ImageOut]


# ── Prediction ────────────────────────────────────────────────────────

class PredictionResult(BaseModel):
    boxes: list[list[float]]
    labels: list[int]
    scores: list[float]
    label_names: list[str]
    heatmap: Optional[str] = None
    model_type: Optional[str] = None
    model_version: Optional[str] = None
```

- [ ] **Step 2: Verify schemas import correctly**

Run: `cd /d/Side_Project/Brain_Tumor_FasterRCNN/backend && python -c "from schemas import DetectionOut, ImageOut, PredictionResult; print('OK')"`
Expected: `OK`

---

### Task 4: Update Detection Router (`backend/routers/predict.py`)

**Files:**
- Modify: `backend/routers/predict.py`

- [ ] **Step 1: Update predict.py**

Changes needed:
1. Add `import hashlib` at the top
2. Import `Detection` from models
3. After inference, save detections as individual rows instead of JSON blob
4. Add model_type and model_version to PredictionResult

Key section to change — the save logic (lines ~227-253). Replace from `# ── Save to disk + DB ──` onward:

```python
    # ── Save to disk + DB ──
    user_dir = os.path.join(cfg.UPLOAD_DIR, str(current_user.id))
    os.makedirs(user_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, stored_name)

    image.save(file_path)

    img = Image.open(io.BytesIO(contents)).convert("RGB")

    db_image = ImageModel(
        user_id=current_user.id,
        original_name=file.filename or "unknown",
        stored_name=stored_name,
        filepath=file_path,
        mime_type=file.content_type or "image/jpeg",
        width=img.width,
        height=img.height,
        file_size=len(contents),
        content_hash=hashlib.sha256(contents).hexdigest(),
        model_type="faster-rcnn",
        model_version="resnet50-fpn-v1",
        threshold=threshold,
    )
    db.add(db_image)
    db.flush()  # get db_image.id

    for box, label, score, label_name in zip(boxes, labels, scores, label_names):
        db.add(Detection(
            image_id=db_image.id,
            label=label,
            label_name=label_name,
            score=score,
            x1=box[0], y1=box[1], x2=box[2], y2=box[3],
        ))
    db.commit()
    db.refresh(db_image)

    return PredictionResult(
        boxes=boxes, labels=labels, scores=scores, label_names=label_names,
        heatmap=heatmap,
        model_type="faster-rcnn",
        model_version="resnet50-fpn-v1",
    )
```

- [ ] **Step 2: Add import for Detection in predict.py**

Add at the top:
```python
import hashlib
```
And change the model import from:
```python
from models import Image as ImageModel, User
```
to:
```python
from models import Image as ImageModel, Detection, User
```

Also fix the duplicate `img = Image.open(io.BytesIO(contents)).convert("RGB")` — the image is loaded earlier at line ~204, so the save block should use that same variable. Let me check the full file...

Actually, looking at the current code:
- Line ~204: `image = Image.open(io.BytesIO(contents)).convert("RGB")`
- Line ~235: `image.save(file_path)` — this uses the PIL Image

So the `img` variable is already `image` from the earlier load. The save block should reuse `image` for width/height. Let me fix:

```python
    db_image = ImageModel(
        ...
        width=image.width,
        height=image.height,
        ...
    )
```

(Remove the redundant `img = Image.open(io.BytesIO(contents)).convert("RGB")` line)

---

### Task 5: Update YOLO Router (`backend/routers/predict_yolo.py`)

**Files:**
- Modify: `backend/routers/predict_yolo.py`

- [ ] **Step 1: Update predict_yolo.py**

Same pattern as predict.py:
1. Add `import hashlib` at the top
2. Import `Detection` from models
3. Replace the save block with individual detection rows
4. Add model info to return

Replace from `# ── Save to disk + DB ──` onward:

```python
    # ── Save to disk + DB ──
    user_dir = os.path.join(cfg.UPLOAD_DIR, str(current_user.id))
    os.makedirs(user_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(user_dir, stored_name)

    image.save(file_path)

    db_image = ImageModel(
        user_id=current_user.id,
        original_name=file.filename or "unknown",
        stored_name=stored_name,
        filepath=file_path,
        mime_type=file.content_type or "image/jpeg",
        width=image.width,
        height=image.height,
        file_size=len(contents),
        content_hash=hashlib.sha256(contents).hexdigest(),
        model_type="yolov8",
        model_version="yolov8n",
        threshold=threshold,
    )
    db.add(db_image)
    db.flush()

    for box, label, score, label_name in zip(boxes, labels, scores, label_names):
        db.add(Detection(
            image_id=db_image.id,
            label=label,
            label_name=label_name,
            score=score,
            x1=box[0], y1=box[1], x2=box[2], y2=box[3],
        ))
    db.commit()

    return PredictionResult(
        boxes=boxes, labels=labels, scores=scores, label_names=label_names,
        model_type="yolov8",
        model_version="yolov8n",
    )
```

- [ ] **Step 2: Add imports**

```python
import hashlib
```
Update model import:
```python
from models import Image as ImageModel, Detection, User
```

---

### Task 6: Update Images Router (`backend/routers/images.py`)

**Files:**
- Modify: `backend/routers/images.py`

- [ ] **Step 1: Update report generation to use detections relationship**

The `generate_report` function reads from `img.detection_result` dict. Change it to read from `img.detections` list of Detection objects.

Replace the report's detection section (around lines 106-157):

```python
    if img.detections:
        detections = img.detections
        scores = [d.score for d in detections]

        meta.append(["Detections", str(len(scores))])
        if scores:
            avg_score = sum(scores) / len(scores)
            meta.append(["Avg Confidence", f"{avg_score * 100:.1f}%"])

    meta_table = Table(meta, colWidths=[120, 300])
    ...
```

And the findings section:

```python
    # ── Detection details table ──
    if img.detections:
        story.append(Paragraph("Findings", styles["Heading2"]))
        story.append(Spacer(1, 3*mm))

        data = [["#", "Label", "Confidence", "Location (x1,y1,x2,y2)"]]
        for i, det in enumerate(img.detections, 1):
            box_str = f"({det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f})"
            data.append([str(i), det.label_name, f"{det.score * 100:.1f}%", box_str])

        findings_table = Table(data, colWidths=[30, 100, 80, 200])
        ...
```

---

### Task 7: Add Token Cleanup to App Startup (`backend/main.py`)

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Update main.py**

Add import and cleanup call:

```python
from database import engine, Base, SessionLocal
from models import purge_expired_tokens
```

In the startup event:
```python
@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        purge_expired_tokens(db)
    os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)
```

---

### Task 8: Generate Alembic Migrations

**Files:**
- Create: `backend/alembic/versions/001_create_initial_tables.py`
- Create: `backend/alembic/versions/002_add_metadata_and_detections.py`
- Create: `backend/alembic/versions/003_add_indexes.py`

- [ ] **Step 1: Generate initial migration (captures current schema)**

First, create a temporary models.py with the OLD schema (keep a copy, restore later). Or better — generate the migration manually since we know the exact schema.

Actually, the best approach: since we're changing models.py to the new design, we need the initial migration to represent the CURRENT schema before our changes. Let me write it manually:

Create `backend/alembic/versions/001_create_initial_tables.py`:
```python
"""Create initial tables (users, token_blacklist, images)

Revision ID: 001
Revises:
Create Date: 2026-06-01
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table("users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("username", sa.String(100), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=True),
        sa.Column("google_id", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_id", "users", ["id"], unique=False)

    op.create_table("token_blacklist",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("jti", sa.String(255), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_token_blacklist_jti", "token_blacklist", ["jti"], unique=True)
    op.create_index("ix_token_blacklist_id", "token_blacklist", ["id"], unique=False)

    op.create_table("images",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("original_name", sa.String(500), nullable=False),
        sa.Column("stored_name", sa.String(255), nullable=False),
        sa.Column("filepath", sa.String(1000), nullable=False),
        sa.Column("mime_type", sa.String(50), nullable=False),
        sa.Column("detection_result", sa.JSON(), nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=True),
        sa.Column("deleted", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_images_id", "images", ["id"], unique=False)


def downgrade() -> None:
    op.drop_table("images")
    op.drop_table("token_blacklist")
    op.drop_table("users")
```

- [ ] **Step 2: Create migration 002 — add metadata, detections table, migrate data**

Create `backend/alembic/versions/002_add_metadata_and_detections.py`:
```python
"""Add image metadata, detections table, migrate JSON data

Revision ID: 002
Revises: 001
"""
import json
from typing import Union, Sequence
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add new columns to images (nullable so existing rows pass)
    op.add_column("images", sa.Column("width", sa.Integer(), nullable=True))
    op.add_column("images", sa.Column("height", sa.Integer(), nullable=True))
    op.add_column("images", sa.Column("file_size", sa.Integer(), nullable=True))
    op.add_column("images", sa.Column("content_hash", sa.String(64), nullable=True))
    op.add_column("images", sa.Column("model_type", sa.String(50), nullable=True))
    op.add_column("images", sa.Column("model_version", sa.String(100), nullable=True))
    op.add_column("images", sa.Column("threshold", sa.Float(), nullable=True))

    # Create detections table
    op.create_table("detections",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("image_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.Integer(), nullable=False),
        sa.Column("label_name", sa.String(100), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("x1", sa.Float(), nullable=False),
        sa.Column("y1", sa.Float(), nullable=False),
        sa.Column("x2", sa.Float(), nullable=False),
        sa.Column("y2", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_detections_id", "detections", ["id"], unique=False)
    op.create_index("ix_detections_image_id", "detections", ["image_id"], unique=False)

    # Migrate existing JSON data from detection_result to detections rows
    conn = op.get_bind()
    rows = conn.execute(
        text("SELECT id, detection_result FROM images WHERE detection_result IS NOT NULL")
    ).fetchall()

    for row in rows:
        img_id, result_json = row
        try:
            result = json.loads(result_json) if isinstance(result_json, str) else result_json
        except (json.JSONDecodeError, TypeError):
            continue

        boxes = result.get("boxes", [])
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        label_names = result.get("label_names", [])

        for i, box in enumerate(boxes):
            label_id = labels[i] if i < len(labels) else 0
            label_name = label_names[i] if i < len(label_names) else f"Class {label_id}"
            score = scores[i] if i < len(scores) else 0.0

            conn.execute(
                text("""INSERT INTO detections (image_id, label, label_name, score, x1, y1, x2, y2)
                       VALUES (:img_id, :label, :label_name, :score, :x1, :y1, :x2, :y2)"""),
                {
                    "img_id": img_id,
                    "label": label_id,
                    "label_name": label_name,
                    "score": score,
                    "x1": box[0], "y1": box[1],
                    "x2": box[2], "y2": box[3],
                },
            )

    # Drop the old JSON column
    op.drop_column("images", "detection_result")

    # Update FK on images.user_id to add CASCADE (SQLite requires recreate)
    # SQLite doesn't support ALTER TABLE ADD CONSTRAINT — 
    # but the FK was already created by Base.metadata.create_all().
    # We'll handle this in the models.py by specifying ondelete="CASCADE"
    # and it'll apply to NEW tables. For existing SQLite DBs, the FK
    # behavior is best-effort (SQLite doesn't enforce FKs unless PRAGMA foreign_keys = ON).


def downgrade() -> None:
    op.add_column("images", sa.Column("detection_result", sa.JSON(), nullable=True))
    op.drop_table("detections")
    op.drop_column("images", "threshold")
    op.drop_column("images", "model_version")
    op.drop_column("images", "model_type")
    op.drop_column("images", "content_hash")
    op.drop_column("images", "file_size")
    op.drop_column("images", "height")
    op.drop_column("images", "width")
```

- [ ] **Step 3: Create migration 003 — add composite indexes**

Create `backend/alembic/versions/003_add_indexes.py`:
```python
"""Add composite indexes for common queries

Revision ID: 003
Revises: 002
"""
from typing import Union, Sequence
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Composite index for history query: WHERE user_id=? AND NOT deleted ORDER BY uploaded_at DESC
    op.create_index("ix_images_user_id_deleted_uploaded_at", "images",
                    ["user_id", "deleted", sa.text("uploaded_at DESC")])

    # Fast soft-delete filter
    op.create_index("ix_images_user_id_deleted", "images", ["user_id", "deleted"])

    # Score DESC index for "best detection" lookup (used by Grad-CAM)
    op.create_index("ix_detections_image_id_score", "detections",
                    ["image_id", sa.text("score DESC")])


def downgrade() -> None:
    op.drop_index("ix_detections_image_id_score")
    op.drop_index("ix_images_user_id_deleted")
    op.drop_index("ix_images_user_id_deleted_uploaded_at")
```

- [ ] **Step 4: Stamp the current DB revision**

Since the database already exists with the current schema, we need to stamp it as being at revision 001 (which matches the current schema):

Run: `cd /d/Side_Project/Brain_Tumor_FasterRCNN/backend && alembic stamp 001`
Expected: `INFO  [alembic.runtime.migration] Context impl SQLiteImpl.`
`INFO  [alembic.runtime.migration] Will assume non-transactional DDL.`
`INFO  [alembic.runtime.migration] Running stamp_revision 001 -> 001`

- [ ] **Step 5: Run migration 002 + 003**

Run: `cd /d/Side_Project/Brain_Tumor_FasterRCNN/backend && alembic upgrade head`
Expected: runs both 002 and 003, output should show:
```
INFO  [alembic.runtime.migration] 002_add_metadata -> 003_add_indexes
```

---

### Task 9: Verify Everything Works

- [ ] **Step 1: Start the backend and test**

Run: `cd /d/Side_Project/Brain_Tumor_FasterRCNN/backend && python -m uvicorn main:app --port 8000`

In another terminal:
```bash
# Health check
curl http://localhost:8000/

# Register a user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","username":"test","password":"test123"}'

# Run a detection
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <token>" \
  -F "file=@test_image.jpg" \
  -F "threshold=0.5"

# List images (should show detections nested)
curl http://localhost:8000/images \
  -H "Authorization: Bearer <token>"
```

Expected: all endpoints work, images list includes `detections` array with individual detection objects.

- [ ] **Step 2: Verify Alembic history**

Run: `cd /d/Side_Project/Brain_Tumor_FasterRCNN/backend && alembic history`
Expected: shows 3 revisions in sequence (001 → 002 → 003)

- [ ] **Step 3: Clean up test data**

Run: `rm backend/brain_tumor.db` (only for cleanup if needed)
