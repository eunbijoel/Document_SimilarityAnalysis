"""일반 텍스트(.txt / .md / .text / .csv) 파서.

폼 피드(\\f) 또는 빈 줄 2개 이상으로 구역을 나누고, 없으면 문단 묶음으로 나눕니다.
"""
from __future__ import annotations

import re
from typing import Optional

from src.models.schemas import ParseResult
from src.parsers.common import build_text_records


def _decode_bytes(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _split_units(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if "\f" in text:
        parts = [p.strip() for p in text.split("\f")]
        return [p for p in parts if p]
    parts = re.split(r"\n\s*\n\s*\n+", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts
    # 단일 블록이면 줄 단위로 적당한 크기로 묶기 (너무 긴 파일 보호)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    chunk_size = 40
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunks.append("\n".join(lines[i : i + chunk_size]))
    return chunks


def parse_text(
    file_name: str,
    file_bytes: bytes,
    min_sentence_length: int = 8,
    min_image_size: Optional[int] = None,  # noqa: ARG001 — 인터페이스 통일
    min_image_width: Optional[int] = None,  # noqa: ARG001
    min_image_height: Optional[int] = None,  # noqa: ARG001
) -> ParseResult:
    try:
        raw = _decode_bytes(file_bytes)
    except Exception as exc:  # noqa: BLE001
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=f"텍스트를 읽을 수 없습니다: {exc}",
        )

    chunks = _split_units(raw)
    if not chunks and raw.strip():
        chunks = [raw.strip()]

    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "txt"
    file_type = ext if ext in {"txt", "md", "text", "csv"} else "txt"

    units = [(i, f"구역 {i}", chunk) for i, chunk in enumerate(chunks, start=1)]
    sentences, pages, _ = build_text_records(
        file_name, file_type, units, min_sentence_length=min_sentence_length
    )
    return ParseResult(
        file_name=file_name,
        success=True,
        sentences=sentences,
        images=[],
        pages=pages,
    )
