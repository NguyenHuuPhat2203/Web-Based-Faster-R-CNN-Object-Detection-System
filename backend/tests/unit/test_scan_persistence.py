"""Unit tests for scan persistence service."""

import io

from PIL import Image as PILImage

from backend.models import User
from backend.services.scan_persistence_service import ScanPersistenceService
from ml.core.entities import Detection, Scan


class TestScanPersistenceService:
    """Scan persistence unit tests."""

    def test_save_scan_result(self, db_session, tmp_path, monkeypatch) -> None:
        """Persist an image and associated detections."""
        from backend.services import scan_persistence_service

        monkeypatch.setattr(
            scan_persistence_service.settings,
            "upload_dir",
            str(tmp_path),
        )

        user = User(email="scan@example.com", username="Scanner")
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        pil_image = PILImage.new("RGB", (32, 32), color=(128, 128, 128))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        contents = buffer.getvalue()

        service = ScanPersistenceService(db_session)
        scan = Scan(
            contents=contents,
            original_name="scan.png",
            content_type="image/png",
            threshold=0.5,
            model_type="faster-rcnn",
            model_version="test",
            detections=[
                Detection(
                    box=[1.0, 2.0, 10.0, 12.0],
                    label=1,
                    score=0.9,
                    label_name="Brain tumor",
                )
            ],
        )
        image = service.save_scan_result(user=user, scan=scan)

        assert image.id is not None
        assert len(image.detections) == 1
        assert image.detections[0].label_name == "Brain tumor"
