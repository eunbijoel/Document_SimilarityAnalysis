"""유사 문장이 있는 PDF 페이지를 PNG로 렌더·저장."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.utils.summary_stats import parse_page_number

_SAFE = re.compile(r"[^\w.\-가-힣]+", re.UNICODE)


def _safe_name(name: str, max_len: int = 40) -> str:
    stem = Path(name).stem
    return _SAFE.sub("_", stem)[:max_len] or "file"


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


def collect_matched_page_pairs(sentence_pairs: list[dict]) -> list[tuple[str, int, str, int]]:
    """
    유사 문장 쌍에서 (파일A, 페이지A, 파일B, 페이지B)를 중복 없이 수집.
    확인이 쉽도록 A/B 페이지가 모두 있는 쌍만 포함한다.
    """
    pairs: set[tuple[str, int, str, int]] = set()
    for p in sentence_pairs:
        pa = parse_page_number(p.get("location_a", ""))
        pb = parse_page_number(p.get("location_b", ""))
        if pa is None or pb is None:
            continue
        fa, fb = p["file_a"], p["file_b"]
        # 순서 정규화 (같은 두 페이지 쌍이 뒤집혀 중복되지 않게)
        if (fa, pa) <= (fb, pb):
            pairs.add((fa, pa, fb, pb))
        else:
            pairs.add((fb, pb, fa, pa))
    return sorted(pairs, key=lambda x: (x[0], x[1], x[2], x[3]))


def _pair_stem(file_a: str, page_a: int, file_b: str, page_b: int) -> str:
    """파일A_p0003=파일B_p0007 형태 공통 파일명 stem."""
    return (
        f"{_safe_name(file_a)}_p{page_a:04d}"
        f"={_safe_name(file_b)}_p{page_b:04d}"
    )


def build_matched_page_screenshots(
    pdf_bytes_by_name: dict[str, bytes],
    page_pairs: list[tuple[str, int, str, int]],
    *,
    dpi: int = 120,
) -> list[dict]:
    """
    유사 문장 페이지 쌍을 PNG로 렌더.

    파일명 예:
      GITCC_..._p0003=스위스_..._p0012__A.png
      GITCC_..._p0003=스위스_..._p0012__B.png

    Returns:
        각 항목은 pair 정보 + side(A/B) + filename + png_bytes
    """
    results: list[dict] = []
    # 같은 (파일,페이지)는 한 번만 렌더하고 재사용
    cache: dict[tuple[str, int], bytes] = {}

    for file_a, page_a, file_b, page_b in page_pairs:
        stem = _pair_stem(file_a, page_a, file_b, page_b)
        pair_label = f"{file_a} p.{page_a} = {file_b} p.{page_b}"

        for side, fname, page in (("A", file_a, page_a), ("B", file_b, page_b)):
            key = (fname, page)
            if key not in cache:
                pdf_bytes = pdf_bytes_by_name.get(fname)
                if not pdf_bytes:
                    continue
                png = render_page_png(pdf_bytes, page, dpi=dpi)
                if not png:
                    continue
                cache[key] = png
            png_bytes = cache.get(key)
            if not png_bytes:
                continue
            out_name = f"{stem}__{side}.png"
            results.append(
                {
                    "pair_label": pair_label,
                    "side": side,
                    "file_name": fname,
                    "page_number": page,
                    "file_a": file_a,
                    "page_a": page_a,
                    "file_b": file_b,
                    "page_b": page_b,
                    "filename": out_name,
                    "png_bytes": png_bytes,
                }
            )
    return results


def screenshots_to_zip(screenshots: list[dict]) -> bytes:
    """스크린샷 목록을 ZIP 바이트로 묶는다."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        used: set[str] = set()
        for item in screenshots:
            name = item["filename"]
            # 동일 파일명 중복 방지
            if name in used:
                continue
            used.add(name)
            zf.writestr(name, item["png_bytes"])
    return buf.getvalue()
