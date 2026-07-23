"""이미지 바이트 -> PIL Image 변환 등 파서 공용 헬퍼.

각 문서 파서(pdf_parser 등)가 추출한 raw image_bytes를 이후 pHash 분석
단계에서 안전하게 열기 위한 공용 함수를 제공합니다.
"""
import io
from typing import Optional

from PIL import Image


def bytes_to_image(image_bytes: bytes) -> Optional[Image.Image]:
    """이미지 바이트를 PIL Image로 변환합니다. 손상된 이미지는 None을 반환합니다."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        return img.convert("RGB")
    except Exception:  # noqa: BLE001
        return None
