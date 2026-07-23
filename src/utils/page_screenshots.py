"""유사 문장이 있는 PDF 페이지를 PNG로 렌더·저장."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


_SAFE = re.compile(r"[^\w.\-가-힣]+", re.UNICODE)


def _safe_name(name: str) -> str:
    stem = Path(name).stem
    return _SAFE.sub("_", stem)[:80] or "file"


def render_page_png(
    pdf_bytes: bytes,
    page_number: int,
    *,
    dpi: int = 120,
) -> Optional[bytes]:
    """1-based page_number 페이지를 PNG 바이트로 렌더."""
    if page_number < 1:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None
    try:
        if page_number > len(doc):
            return None
        page = doc[page_number - 1]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return None
    finally:
        doc.close()


def build_matched_page_screenshots(
    pdf_bytes_by_name: dict[str, bytes],
    page_keys: list[tuple[str, int]],
    *,
    dpi: int = 120,
) -> list[dict]:
    """
    (file_name, page_number) 목록을 PNG로 렌더.

    Returns:
        [{file_name, page_number, filename, png_bytes}, ...]
    """
    results: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for file_name, page_number in page_keys:
        key = (file_name, page_number)
        if key in seen:
            continue
        seen.add(key)
        pdf_bytes = pdf_bytes_by_name.get(file_name)
        if not pdf_bytes:
            continue
        png = render_page_png(pdf_bytes, page_number, dpi=dpi)
        if not png:
            continue
        out_name = f"{_safe_name(file_name)}_p{page_number:04d}.png"
        results.append(
            {
                "file_name": file_name,
                "page_number": page_number,
                "filename": out_name,
                "png_bytes": png,
            }
        )
    return results


def screenshots_to_zip(screenshots: list[dict]) -> bytes:
    """스크린샷 목록을 ZIP 바이트로 묶는다."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in screenshots:
            zf.writestr(item["filename"], item["png_bytes"])
    return buf.getvalue()
