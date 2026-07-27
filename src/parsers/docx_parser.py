"""DOCX에서 텍스트(문장)·이미지·논리 페이지를 추출합니다.

실제 인쇄 페이지 대신 강제 페이지 나누기 / 섹션 단위로 PageRecord를 만듭니다.
"""
from __future__ import annotations

import io
from typing import Optional

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.models.schemas import ParseResult
from src.parsers.common import build_text_records, maybe_image_record, resolve_image_mins
from src.parsers.image_parser import bytes_to_image


def _iter_block_items(document: Document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _paragraph_has_page_break(paragraph: Paragraph) -> bool:
    for run in paragraph.runs:
        for br in run._element.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def _table_text(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_docx(
    file_name: str,
    file_bytes: bytes,
    min_sentence_length: int = 8,
    min_image_size: Optional[int] = None,
    min_image_width: Optional[int] = None,
    min_image_height: Optional[int] = None,
) -> ParseResult:
    width_min, height_min = resolve_image_mins(min_image_size, min_image_width, min_image_height)

    try:
        document = Document(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=f"DOCX를 열 수 없습니다: {exc}",
        )

    units: list[tuple[int, str, str]] = []
    page_number = 1
    buffer: list[str] = []

    def flush() -> None:
        nonlocal page_number, buffer
        text = "\n".join(buffer).strip()
        if text:
            units.append((page_number, f"구역 {page_number}", text))
            page_number += 1
        buffer = []

    try:
        for block in _iter_block_items(document):
            if isinstance(block, Paragraph):
                text = (block.text or "").strip()
                if text:
                    buffer.append(text)
                if _paragraph_has_page_break(block):
                    flush()
            elif isinstance(block, Table):
                text = _table_text(block)
                if text:
                    buffer.append(text)
        flush()
        if not units:
            # 빈 문서가 아니면 최소한 전체 텍스트라도 시도
            full = "\n".join(p.text for p in document.paragraphs if p.text and p.text.strip())
            if full.strip():
                units.append((1, "구역 1", full))
    except Exception as exc:  # noqa: BLE001
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=f"DOCX 텍스트 추출 실패: {exc}",
        )

    sentences, pages, _ = build_text_records(
        file_name, "docx", units, min_sentence_length=min_sentence_length
    )

    images = []
    try:
        image_pos = 0
        for rel in document.part.rels.values():
            if "image" not in getattr(rel, "reltype", ""):
                continue
            try:
                blob = rel.target_part.blob
            except Exception:  # noqa: BLE001
                continue
            pil = bytes_to_image(blob)
            if pil is None:
                continue
            width, height = pil.size
            rec = maybe_image_record(
                file_name=file_name,
                location="문서 이미지",
                image_id=f"{file_name}::img{image_pos}",
                image_bytes=blob,
                width=width,
                height=height,
                width_min=width_min,
                height_min=height_min,
            )
            if rec:
                images.append(rec)
                image_pos += 1
    except Exception:  # noqa: BLE001
        pass

    return ParseResult(
        file_name=file_name,
        success=True,
        sentences=sentences,
        images=images,
        pages=pages,
    )
