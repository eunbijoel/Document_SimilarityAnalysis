"""HWPX(한컴 오피스 XML) 파서 — ZIP + section*.xml에서 텍스트 추출."""
from __future__ import annotations

import io
import re
import zipfile
from typing import Optional
from xml.etree import ElementTree as ET

from src.models.schemas import ParseResult
from src.parsers.common import build_text_records, maybe_image_record, resolve_image_mins
from src.parsers.image_parser import bytes_to_image

_SECTION_NAME = re.compile(r"Contents/section\d+\.xml$", re.IGNORECASE)


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _paragraph_text(para: ET.Element) -> str:
    parts: list[str] = []
    for node in para.iter():
        if _local(node.tag) == "t" and node.text:
            parts.append(node.text)
        elif _local(node.tag) in {"lineBreak", "lb"}:
            parts.append("\n")
        elif _local(node.tag) == "tab":
            parts.append("\t")
    return "".join(parts).strip()


def _section_paragraphs(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    paras: list[str] = []
    for node in root.iter():
        if _local(node.tag) == "p":
            text = _paragraph_text(node)
            if text:
                paras.append(text)
    return paras


def parse_hwpx(
    file_name: str,
    file_bytes: bytes,
    min_sentence_length: int = 8,
    min_image_size: Optional[int] = None,
    min_image_width: Optional[int] = None,
    min_image_height: Optional[int] = None,
) -> ParseResult:
    width_min, height_min = resolve_image_mins(min_image_size, min_image_width, min_image_height)

    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=f"HWPX(ZIP)를 열 수 없습니다: {exc}",
        )

    try:
        section_names = sorted(n for n in zf.namelist() if _SECTION_NAME.search(n))
        if not section_names:
            return ParseResult(
                file_name=file_name,
                success=False,
                error_message="HWPX에 Contents/section*.xml이 없습니다.",
            )

        units: list[tuple[int, str, str]] = []
        for idx, name in enumerate(section_names, start=1):
            paras = _section_paragraphs(zf.read(name))
            text = "\n".join(paras)
            units.append((idx, f"섹션 {idx}", text))

        images = []
        image_pos = 0
        for name in zf.namelist():
            lower = name.lower()
            if not lower.startswith("bindata/") and "/bindata/" not in lower:
                continue
            if not lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff")):
                continue
            try:
                blob = zf.read(name)
            except Exception:  # noqa: BLE001
                continue
            pil = bytes_to_image(blob)
            if pil is None:
                continue
            w, h = pil.size
            rec = maybe_image_record(
                file_name=file_name,
                location="문서 이미지",
                image_id=f"{file_name}::img{image_pos}",
                image_bytes=blob,
                width=w,
                height=h,
                width_min=width_min,
                height_min=height_min,
            )
            if rec:
                images.append(rec)
                image_pos += 1
    except Exception as exc:  # noqa: BLE001
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=f"HWPX 추출 실패: {exc}",
        )
    finally:
        zf.close()

    sentences, pages, _ = build_text_records(
        file_name, "hwpx", units, min_sentence_length=min_sentence_length
    )
    return ParseResult(
        file_name=file_name,
        success=True,
        sentences=sentences,
        images=images,
        pages=pages,
    )
