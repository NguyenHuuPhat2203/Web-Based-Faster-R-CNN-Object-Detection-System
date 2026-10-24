"""PDF report generation for detection results."""

from __future__ import annotations

import io
import os
from datetime import UTC, datetime

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..models import Detection, Image

_BOX_COLOR = (16, 185, 129)


def reported_detections(image: Image) -> list[Detection]:
    """Detections at or above the scan's threshold.

    v2 predictions persist everything above a low score floor so the client
    slider can filter; the report must show what the user saw at the saved
    threshold, not the raw floor.
    """
    threshold = image.threshold if image.threshold is not None else 0.0
    return [d for d in image.detections or [] if d.score >= threshold]


def annotated_preview_bytes(image: Image) -> bytes:
    """Return PNG bytes of the stored scan with reported boxes drawn."""
    with PILImage.open(image.filepath) as source:
        pil = ImageOps.exif_transpose(source).convert("RGB")
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    for detection in reported_detections(image):
        draw.rectangle(
            [detection.x1, detection.y1, detection.x2, detection.y2],
            outline=_BOX_COLOR,
            width=3,
        )
        label = f"{detection.label_name}: {detection.score * 100:.1f}%"
        text_y = max(0, detection.y1 - 12)
        draw.text((detection.x1 + 4, text_y), label, fill=(255, 255, 255), font=font)
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return buffer.getvalue()


def meta_rows(image: Image) -> list[list[str]]:
    """Build the report metadata table rows from a stored scan."""
    rows = [
        ["Scan ID", f"IMG-{image.id:06d}"],
        ["Original File", image.original_name],
        ["Uploaded", image.uploaded_at.strftime("%Y-%m-%d %H:%M UTC")],
        ["Report Generated", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")],
    ]
    if image.model_type:
        rows.append(["Model", str(image.model_type)])
    if image.model_version:
        rows.append(["Version", str(image.model_version)])
    if image.threshold is not None:
        rows.append(["Threshold", f"{float(image.threshold):.2f}"])
    detections = reported_detections(image)
    if detections:
        scores = [detection.score for detection in detections]
        rows.append(["Detections", str(len(scores))])
        avg_score = sum(scores) / len(scores)
        rows.append(["Avg Confidence", f"{avg_score * 100:.1f}%"])
    return rows


class ReportService:
    """Build downloadable PDF reports for scan results."""

    def generate_pdf(self, image: Image) -> bytes:
        """Generate a PDF report for a stored image.

        Args:
            image: Image ORM instance with detections loaded.

        Returns:
            PDF document bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
        )
        styles = getSampleStyleSheet()
        story: list = []

        story.append(Paragraph("Brain Tumor Detection Report", styles["Title"]))
        story.append(Spacer(1, 6 * mm))

        meta_table = Table(meta_rows(image), colWidths=[120, 300])
        meta_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1e293b")),
                    ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#334155")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 0),
                        (-1, -1),
                        [colors.HexColor("#1e293b"), colors.HexColor("#0f172a")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(meta_table)
        story.append(Spacer(1, 6 * mm))

        detections = reported_detections(image)
        if detections:
            story.append(Paragraph("Findings", styles["Heading2"]))
            story.append(Spacer(1, 3 * mm))

            data = [["#", "Label", "Confidence", "Location (x1,y1,x2,y2)"]]
            for index, detection in enumerate(detections, 1):
                box_str = (
                    f"({detection.x1:.0f}, {detection.y1:.0f}, "
                    f"{detection.x2:.0f}, {detection.y2:.0f})"
                )
                data.append(
                    [
                        str(index),
                        detection.label_name,
                        f"{detection.score * 100:.1f}%",
                        box_str,
                    ]
                )

            findings_table = Table(data, colWidths=[30, 100, 80, 200])
            findings_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ALIGN", (0, 0), (0, -1), "CENTER"),
                        ("ALIGN", (2, 0), (2, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#334155")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.HexColor("#1e293b"), colors.HexColor("#0f172a")],
                        ),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            story.append(findings_table)
            story.append(Spacer(1, 6 * mm))

        if image.filepath and os.path.exists(image.filepath):
            story.append(Paragraph("Annotated Image", styles["Heading2"]))
            story.append(Spacer(1, 3 * mm))

            preview = io.BytesIO(annotated_preview_bytes(image))
            with PILImage.open(preview) as pil_image:
                max_w = 460.0
                width, height = float(pil_image.size[0]), float(pil_image.size[1])
            if width > max_w:
                ratio = max_w / width
                width, height = max_w, height * ratio
            preview.seek(0)
            story.append(RLImage(preview, width=width, height=height))
            story.append(Spacer(1, 4 * mm))

        story.append(Spacer(1, 10 * mm))
        story.append(
            Paragraph(
                "<i>This report was automatically generated. "
                "Findings should be reviewed by a qualified medical professional.</i>",
                styles["Italic"],
            )
        )

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
