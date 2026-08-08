"""Image history and report HTTP routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from ...core.database import get_db
from ...models import User
from ...schemas.image import ImageList, ImageOut
from ...services.image_service import ImageService
from ...services.report_service import ReportService
from ..dependencies import get_current_user

router = APIRouter(prefix="/images", tags=["images"])


@router.get("", response_model=ImageList)
def list_images(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> ImageList:
    """List all active images for the authenticated user."""
    images = ImageService(db).list_for_user(current_user)
    return ImageList(images=[ImageOut.model_validate(image) for image in images])


@router.get("/{image_id}", response_class=FileResponse)
def get_image(
    image_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> FileResponse:
    """Return the stored image file for an owned image."""
    image = ImageService(db).get_owned_image(current_user, image_id)
    return FileResponse(str(image.filepath), media_type=str(image.mime_type))


@router.delete("/{image_id}")
def delete_image(
    image_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> dict[str, str]:
    """Soft-delete an owned image."""
    ImageService(db).soft_delete(current_user, image_id)
    return {"message": "Image deleted"}


@router.get("/{image_id}/report", response_class=Response)
def generate_report(
    image_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> Response:
    """Generate a PDF report for an owned detection result."""
    image = ImageService(db).get_owned_image(current_user, image_id)
    pdf_bytes = ReportService().generate_pdf(image)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=report-{image.id}.pdf",
        },
    )
