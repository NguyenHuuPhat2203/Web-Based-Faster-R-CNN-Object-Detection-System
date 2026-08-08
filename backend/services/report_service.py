"""PDF report generation for detection results."""

from __future__ import annotations

import io
import os
from datetime import UTC, datetime

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from ..models import Image


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

        meta = [
            ["Patient ID", f"IMG-{image.id:06d}"],
            ["Original File", image.original_name],
            ["Uploaded", image.uploaded_at.strftime("%Y-%m-%d %H:%M UTC")],
            [
                "Report Generated",
                datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            ],
        ]

        if image.detections:
            scores = [detection.score for detection in image.detections]
            meta.append(["Detections", str(len(scores))])
            if scores:
                avg_score = sum(scores) / len(scores)
                meta.append(["Avg Confidence", f"{avg_score * 100:.1f}%"])

        meta_table = Table(meta, colWidths=[120, 300])
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

        if image.detections:
            story.append(Paragraph("Findings", styles["Heading2"]))
            story.append(Spacer(1, 3 * mm))

            data = [["#", "Label", "Confidence", "Location (x1,y1,x2,y2)"]]
            for index, detection in enumerate(image.detections, 1):
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

            pil_image = PILImage.open(image.filepath)
            max_w = 460.0
            width, height = float(pil_image.size[0]), float(pil_image.size[1])
            if width > max_w:
                ratio = max_w / width
                width, height = max_w, height * ratio

            story.append(RLImage(image.filepath, width=width, height=height))
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
