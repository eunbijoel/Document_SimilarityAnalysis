"""HWP 5.x (OLE 복합문서) 텍스트 추출.

한/글 바이너리 BodyText 섹션의 HWPTAG_PARA_TEXT(67) 레코드를 순회합니다.
완전 호환은 어렵지만, 일반적인 한글 문서 본문 추출에는 충분합니다.
이미지는 지원하지 않습니다 (페이지 PNG도 PDF만 가능).
"""
from __future__ import annotations

import io
import struct
import zlib
from typing import Optional

import olefile

from src.models.schemas import ParseResult
from src.parsers.common import build_text_records

# HWP 5.x record
HWPTAG_PARA_TEXT = 67


def _is_compressed(header: bytes) -> bool:
    if len(header) < 38:
        return True
    # FileHeader flags: bit 0 = compressed (흔한 구현)
    return bool(header[36] & 0x01)


def _decompress(data: bytes) -> bytes:
    for wbits in (-15, 15):
        try:
            return zlib.decompress(data, wbits)
        except zlib.error:
            continue
    return data


def _decode_para_text(payload: bytes) -> str:
    """UTF-16LE 본문. 인라인 컨트롤(유니코드 private area 등)은 건너뜁니다."""
    if len(payload) < 2:
        return ""
    chars: list[str] = []
    i = 0
    while i + 1 < len(payload):
        code = struct.unpack_from("<H", payload, i)[0]
        i += 2
        if code in (0x000A, 0x000D):  # line / para break markers often appear as control
            chars.append("\n")
            continue
        if code < 32:
            # 일부 인라인 객체: 뒤따르는 정보 길이가 가변 → 안전히 스킵
            if code in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11):
                # 흔한 패턴: code 뒤 8바이트 메타. 실패해도 다음 문자로 진행.
                if i + 8 <= len(payload):
                    i += 8
            continue
        if 0xE000 <= code <= 0xF8FF:
            continue
        try:
            chars.append(chr(code))
        except ValueError:
            continue
    return "".join(chars)


def _extract_section_text(data: bytes) -> str:
    texts: list[str] = []
    i = 0
    n = len(data)
    while i + 4 <= n:
        header = struct.unpack_from("<I", data, i)[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        i += 4
        if size == 0xFFF:
            if i + 4 > n:
                break
            size = struct.unpack_from("<I", data, i)[0]
            i += 4
        if i + size > n:
            break
        payload = data[i : i + size]
        i += size
        if tag_id == HWPTAG_PARA_TEXT:
            para = _decode_para_text(payload).strip()
            if para:
                texts.append(para)
    return "\n".join(texts)


def _hwp_to_units(file_bytes: bytes) -> list[tuple[int, str, str]]:
    ole = olefile.OleFileIO(io.BytesIO(file_bytes))
    try:
        compressed = True
        if ole.exists("FileHeader"):
            compressed = _is_compressed(ole.openstream("FileHeader").read())

        section_entries = sorted(
            entry for entry in ole.listdir() if entry and entry[0] == "BodyText"
        )
        if not section_entries:
            raise ValueError("BodyText 섹션을 찾을 수 없습니다.")

        units: list[tuple[int, str, str]] = []
        for idx, entry in enumerate(section_entries, start=1):
            raw = ole.openstream(entry).read()
            if compressed:
                raw = _decompress(raw)
            text = _extract_section_text(raw)
            units.append((idx, f"섹션 {idx}", text))
        return units
    finally:
        ole.close()


def parse_hwp(
    file_name: str,
    file_bytes: bytes,
    min_sentence_length: int = 8,
    min_image_size: Optional[int] = None,  # noqa: ARG001
    min_image_width: Optional[int] = None,  # noqa: ARG001
    min_image_height: Optional[int] = None,  # noqa: ARG001
) -> ParseResult:
    # ZIP 기반 HWPX가 .hwp로 잘못 올라온 경우
    if file_bytes[:2] == b"PK":
        from src.parsers.hwpx_parser import parse_hwpx

        return parse_hwpx(
            file_name,
            file_bytes,
            min_sentence_length=min_sentence_length,
            min_image_size=min_image_size,
            min_image_width=min_image_width,
            min_image_height=min_image_height,
        )

    try:
        if not olefile.isOleFile(io.BytesIO(file_bytes)):
            return ParseResult(
                file_name=file_name,
                success=False,
                error_message="HWP OLE 형식이 아닙니다. HWPX로 저장 후 다시 시도해 주세요.",
            )
        units = _hwp_to_units(file_bytes)
    except Exception as exc:  # noqa: BLE001
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=(
                f"HWP 텍스트 추출 실패: {exc}. "
                "가능하면 한글에서 HWPX로 저장한 뒤 업로드해 주세요."
            ),
        )

    if not any(t.strip() for _, _, t in units):
        return ParseResult(
            file_name=file_name,
            success=False,
            error_message=(
                "HWP에서 본문 텍스트를 찾지 못했습니다. "
                "HWPX로 저장하거나 PDF로 변환 후 업로드해 주세요."
            ),
        )

    sentences, pages, _ = build_text_records(
        file_name, "hwp", units, min_sentence_length=min_sentence_length
    )
    return ParseResult(
        file_name=file_name,
        success=True,
        sentences=sentences,
        images=[],
        pages=pages,
    )
