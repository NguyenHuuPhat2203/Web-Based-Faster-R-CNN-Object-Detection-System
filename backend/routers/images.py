"""CRUD routes for user‑owned images + PDF report generation."""

from __future__ import annotations

import io
import os
from datetime import timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from database import get_db
from models import Image as ImageModel, User
from schemas import ImageList
from dependencies import get_current_user

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=ImageList)
def list_images(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    images = (
        db.query(ImageModel)
        .filter(ImageModel.user_id == current_user.id, ~ImageModel.deleted)
        .order_by(ImageModel.uploaded_at.desc())
        .all()
    )
    return ImageList(images=images)


@router.get("/{image_id}", response_class=FileResponse)
def get_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    img = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not img or img.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    if img.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your image")
    return FileResponse(img.filepath, media_type=img.mime_type)


@router.delete("/{image_id}")
def delete_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    img = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not img or img.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    if img.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your image")

    # Soft delete
    img.deleted = True
    db.commit()
    return {"message": "Image deleted"}


@router.get("/{image_id}/report", response_class=Response)
def generate_report(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a PDF report for a detection result."""
    img = db.query(ImageModel).filter(ImageModel.id == image_id).first()
    if not img or img.deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")
    if img.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not your image")

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle,
    )
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []

    # ── Title ─────────────────────────────────────────────────────────
    story.append(Paragraph("Brain Tumor Detection Report", styles["Title"]))
    story.append(Spacer(1, 6*mm))

    # ── Metadata ──────────────────────────────────────────────────────
    meta = [
        ["Patient ID", f"IMG-{img.id:06d}"],
        ["Original File", img.original_name],
        ["Uploaded", img.uploaded_at.strftime("%Y-%m-%d %H:%M UTC")],
        ["Report Generated", __import__("datetime").datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
    ]

    if img.detection_result:
        det = img.detection_result
        scores = det.get("scores", [])

        meta.append(["Detections", str(len(scores))])
        if scores:
            avg_score = sum(scores) / len(scores)
            meta.append(["Avg Confidence", f"{avg_score * 100:.1f}%"])

    meta_table = Table(meta, colWidths=[120, 300])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#334155")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#1e293b"), colors.HexColor("#0f172a")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6*mm))

    # ── Detection details table ───────────────────────────────────────
    if img.detection_result and img.detection_result.get("boxes"):
        det = img.detection_result
        story.append(Paragraph("Findings", styles["Heading2"]))
        story.append(Spacer(1, 3*mm))

        data = [["#", "Label", "Confidence", "Location (x1,y1,x2,y2)"]]
        for i, (box, score) in enumerate(zip(det["boxes"], det["scores"]), 1):
            label = (det.get("label_names") or [""] * len(det["boxes"]))[i - 1] or f"Class {det['labels'][i-1]}"
            box_str = f"({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})"
            data.append([str(i), label, f"{score * 100:.1f}%", box_str])

        findings_table = Table(data, colWidths=[30, 100, 80, 200])
        findings_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#334155")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#1e293b"), colors.HexColor("#0f172a")]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(findings_table)
        story.append(Spacer(1, 6*mm))

    # ── Annotated image ───────────────────────────────────────────────
    if img.filepath and os.path.exists(img.filepath):
        story.append(Paragraph("Annotated Image", styles["Heading2"]))
        story.append(Spacer(1, 3*mm))

        # Load the image, scale to fit within A4 width
        pil_img = __import__("PIL.Image", fromlist=["Image"]).open(img.filepath)
        max_w = 460  # points
        w, h = pil_img.size
        if w > max_w:
            ratio = max_w / w
            w, h = max_w, h * ratio

        story.append(RLImage(img.filepath, width=w, height=h))
        story.append(Spacer(1, 4*mm))

    # ── Footer ────────────────────────────────────────────────────────
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph(
        "<i>This report was automatically generated. "
        "Findings should be reviewed by a qualified medical professional.</i>",
        styles["Italic"],
    ))

    doc.build(story)
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=report-{img.id}.pdf",
        },
    )
