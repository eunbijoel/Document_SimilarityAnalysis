"""파서 공용: 페이지 텍스트 → 문장/페이지 레코드 변환."""
from __future__ import annotations

from src.models.schemas import ImageRecord, PageRecord, SentenceRecord
from src.utils.config import DEFAULT_MIN_IMAGE_HEIGHT, DEFAULT_MIN_IMAGE_WIDTH
from src.utils.text_processing import (
    is_valid_sentence,
    make_sentence_id,
    normalize_whitespace,
    split_into_sentences,
)


def resolve_image_mins(
    min_image_size: int | None,
    min_image_width: int | None,
    min_image_height: int | None,
) -> tuple[int, int]:
    width_min = min_image_width if min_image_width is not None else (
        min_image_size if min_image_size is not None else DEFAULT_MIN_IMAGE_WIDTH
    )
    height_min = min_image_height if min_image_height is not None else (
        min_image_size if min_image_size is not None else DEFAULT_MIN_IMAGE_HEIGHT
    )
    return width_min, height_min


def build_text_records(
    file_name: str,
    file_type: str,
    units: list[tuple[int, str, str]],
    *,
    min_sentence_length: int = 8,
    sentence_index_start: int = 0,
) -> tuple[list[SentenceRecord], list[PageRecord], int]:
    """단위 텍스트 목록을 SentenceRecord / PageRecord로 변환합니다.

    Args:
        units: (page_number, location, raw_text) 목록
    """
    sentences: list[SentenceRecord] = []
    pages: list[PageRecord] = []
    sentence_index = sentence_index_start

    for page_number, location, raw_text in units:
        page_text = normalize_whitespace(raw_text or "")
        if page_text:
            pages.append(
                PageRecord(
                    file_name=file_name,
                    page_number=page_number,
                    text=page_text,
                    location=location,
                )
            )
        for raw_sentence in split_into_sentences(raw_text or ""):
            normalized = normalize_whitespace(raw_sentence)
            if not is_valid_sentence(normalized, min_length=min_sentence_length):
                continue
            sentences.append(
                SentenceRecord(
                    file_name=file_name,
                    file_type=file_type,
                    location=location,
                    text=raw_sentence,
                    normalized_text=normalized,
                    sentence_id=make_sentence_id(file_name, sentence_index),
                )
            )
            sentence_index += 1

    return sentences, pages, sentence_index


def maybe_image_record(
    *,
    file_name: str,
    location: str,
    image_id: str,
    image_bytes: bytes,
    width: int,
    height: int,
    width_min: int,
    height_min: int,
) -> ImageRecord | None:
    if not image_bytes or width < width_min or height < height_min:
        return None
    return ImageRecord(
        file_name=file_name,
        location=location,
        image_id=image_id,
        width=width,
        height=height,
        image_bytes=image_bytes,
    )
