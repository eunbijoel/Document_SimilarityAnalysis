"""PDF 파일에서 텍스트(문장)와 이미지를 추출하는 파서.

PyMuPDF(fitz)를 사용합니다. 페이지 단위로 텍스트를 추출한 뒤 문장으로 분리하고,
각 페이지에 포함된 임베디드 이미지를 추출합니다.
"""
from __future__ import annotations

from typing import Optional

import fitz  # PyMuPDF

from src.models.schemas import ParseResult, SentenceRecord, ImageRecord, PageRecord
from src.utils.config import DEFAULT_MIN_IMAGE_WIDTH, DEFAULT_MIN_IMAGE_HEIGHT
from src.utils.text_processing import (
    normalize_whitespace,
    split_into_sentences,
    is_valid_sentence,
    make_sentence_id,
)


def parse_pdf(
    file_name: str,
    file_bytes: bytes,
    min_sentence_length: int = 8,
    min_image_size: Optional[int] = None,
    min_image_width: Optional[int] = None,
    min_image_height: Optional[int] = None,
) -> ParseResult:
    """PDF 바이트를 받아 페이지/문장/이미지 레코드를 추출합니다.

    Args:
        file_name: 원본 파일명 (표시용)
        file_bytes: 파일 바이너리 내용
        min_sentence_length: 이 길이 미만의 문장은 제외
        min_image_size: (하위 호환) 가로·세로 공통 최소값
        min_image_width / min_image_height: 로고·아이콘 제외용 (초안 기본 180×120)
    """
    # 가로/세로 각각 적용 (초안과 동일). 단일 값이 오면 둘 다에 사용.
    width_min = min_image_width if min_image_width is not None else (
        min_image_size if min_image_size is not None else DEFAULT_MIN_IMAGE_WIDTH
    )
    height_min = min_image_height if min_image_height is not None else (
        min_image_size if min_image_size is not None else DEFAULT_MIN_IMAGE_HEIGHT
    )

    sentences: list[SentenceRecord] = []
    images: list[ImageRecord] = []
    pages: list[PageRecord] = []
    sentence_index = 0

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=f"PDF를 열 수 없습니다: {exc}",
        )

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            location = f"페이지 {page_number}"

            # --- 텍스트 추출 ---
            try:
                raw_text = page.get_text("text") or ""
            except Exception:  # noqa: BLE001
                raw_text = ""

            page_text = normalize_whitespace(raw_text)
            if page_text:
                pages.append(
                    PageRecord(
                        file_name=file_name,
                        page_number=page_number,
                        text=page_text,
                        location=location,
                    )
                )

            for raw_sentence in split_into_sentences(raw_text):
                normalized = normalize_whitespace(raw_sentence)
                if not is_valid_sentence(normalized, min_length=min_sentence_length):
                    continue
                sentences.append(
                    SentenceRecord(
                        file_name=file_name,
                        file_type="pdf",
                        location=location,
                        text=raw_sentence,
                        normalized_text=normalized,
                        sentence_id=make_sentence_id(file_name, sentence_index),
                    )
                )
                sentence_index += 1

            # --- 이미지 추출 ---
            try:
                image_list = page.get_images(full=True)
            except Exception:  # noqa: BLE001
                image_list = []

            for image_pos, image_info in enumerate(image_list):
                xref = image_info[0]
                try:
                    base_image = doc.extract_image(xref)
                except Exception:  # noqa: BLE001
                    continue
                width = int(base_image.get("width", 0) or 0)
                height = int(base_image.get("height", 0) or 0)
                if width < width_min or height < height_min:
                    continue
                image_bytes = base_image.get("image", b"")
                if not image_bytes:
                    continue
                images.append(
                    ImageRecord(
                        file_name=file_name,
                        location=location,
                        image_id=f"{file_name}::p{page_number}::img{image_pos}",
                        width=width,
                        height=height,
                        image_bytes=image_bytes,
                    )
                )
    finally:
        doc.close()

    return ParseResult(
        file_name=file_name,
        success=True,
        sentences=sentences,
        images=images,
        pages=pages,
    )
