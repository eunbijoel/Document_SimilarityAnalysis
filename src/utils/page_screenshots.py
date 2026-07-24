"""유사 문장이 있는 PDF 페이지를 PNG로 렌더·저장 (매칭 문장 하이라이트 포함)."""

from __future__ import annotations

import io
import re
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.utils.summary_stats import parse_page_number

_SAFE = re.compile(r"[^\w.\-가-힣]+", re.UNICODE)
_WS = re.compile(r"\s+")


def _safe_name(name: str, max_len: int = 40) -> str:
    stem = Path(name).stem
    return _SAFE.sub("_", stem)[:max_len] or "file"


def _norm(text: str) -> str:
    return _WS.sub(" ", (text or "").strip())


def _search_queries(text: str) -> list[str]:
    """PDF search_for 용 후보 문자열 (긴 문장·공백 차이 대비)."""
    t = _norm(text)
    if not t:
        return []
    queries: list[str] = [t]
    compact = t.replace(" ", "")
    if compact != t and len(compact) >= 8:
        queries.append(compact)

    # 긴 문장은 앞·뒤·중간 조각을 추가로 시도
    if len(t) > 60:
        queries.append(t[:50])
        queries.append(t[-50:])
        mid = len(t) // 2
        queries.append(t[max(0, mid - 25) : mid + 25])

    # 개조식 앞머리(ㅇ, -) 제거 후 재시도
    stripped = re.sub(r"^[ㅇ\-–•·]\s*", "", t)
    if stripped and stripped != t:
        queries.append(stripped)
        if len(stripped) > 60:
            queries.append(stripped[:50])

    # 중복 제거, 너무 짧은 쿼리 제외
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = q.strip()
        if len(q) < 6 or q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def _find_rects(page: fitz.Page, texts: list[str]) -> list[fitz.Rect]:
    """페이지에서 문장 텍스트에 해당하는 사각형들을 찾는다."""
    rects: list[fitz.Rect] = []
    seen: set[tuple] = set()

    def _add(r: fitz.Rect) -> None:
        key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
        if key in seen or r.is_empty:
            return
        seen.add(key)
        rects.append(fitz.Rect(r))

    # 페이지 단어 목록 (한 번만)
    try:
        words = page.get_text("words") or []
    except Exception:
        words = []

    for text in texts:
        found_any = False
        for q in _search_queries(text):
            try:
                hits = page.search_for(q, quads=False) or []
            except Exception:
                hits = []
            if hits:
                found_any = True
                for r in hits:
                    _add(fitz.Rect(r))
                break
        if found_any:
            continue

        # 공백/nbsp 차이나 search_for 실패 시: 단어 연속 매칭
        for r in _match_word_spans(words, text):
            found_any = True
            _add(r)
    return rects


def _match_word_spans(words: list, text: str) -> list[fitz.Rect]:
    """get_text('words') 결과에서 정규화 문자열이 이어지는 구간을 찾는다."""
    target = _norm(text).replace(" ", "").replace("\xa0", "")
    if len(target) < 6 or not words:
        return []

    # word tuple: (x0, y0, x1, y1, "word", block, line, word_no)
    tokens = []
    for w in words:
        raw = str(w[4]).replace("\xa0", " ").strip()
        if not raw:
            continue
        tokens.append((fitz.Rect(w[0], w[1], w[2], w[3]), raw.replace(" ", "")))

    if not tokens:
        return []

    concat = "".join(t[1] for t in tokens)
    # 너무 긴 타깃은 앞 80자만
    needle = target[:80]
    idx = concat.find(needle)
    if idx < 0 and len(target) > 30:
        needle = target[:40]
        idx = concat.find(needle)
    if idx < 0:
        return []

    end = idx + len(needle)
    # 문자 오프셋 → 토큰 구간
    pos = 0
    start_i = end_i = None
    for i, (_, tok) in enumerate(tokens):
        nxt = pos + len(tok)
        if start_i is None and nxt > idx:
            start_i = i
        if start_i is not None and nxt >= end:
            end_i = i
            break
        pos = nxt
    if start_i is None or end_i is None:
        return []

    # 같은 줄끼리 묶어서 rect union
    from collections import defaultdict

    by_line: dict[int, list[fitz.Rect]] = defaultdict(list)
    for i in range(start_i, end_i + 1):
        # 원본 words 인덱스와 tokens 인덱스가 어긋날 수 있어 line은 y로 근사
        r = tokens[i][0]
        line_key = int(round(r.y0))
        by_line[line_key].append(r)

    out: list[fitz.Rect] = []
    for rs in by_line.values():
        u = rs[0]
        for r in rs[1:]:
            u |= r
        out.append(u)
    return out


def _apply_highlights(page: fitz.Page, rects: list[fitz.Rect]) -> int:
    """노란색 하이라이트 주석을 그리고, 적용 개수를 반환."""
    count = 0
    for r in rects:
        if r.is_empty or r.width < 1 or r.height < 1:
            continue
        try:
            annot = page.add_highlight_annot(r)
            annot.set_colors(stroke=(1.0, 0.92, 0.2))
            annot.set_opacity(0.55)
            annot.update()
            count += 1
        except Exception:
            # 하이라이트 실패 시 반투명 사각형으로 대체
            try:
                shape = page.new_shape()
                shape.draw_rect(r)
                shape.finish(color=(0.95, 0.75, 0.0), fill=(1.0, 0.95, 0.2), fill_opacity=0.35, width=0.5)
                shape.commit()
                count += 1
            except Exception:
                continue
    return count


def render_page_png(
    pdf_bytes: bytes,
    page_number: int,
    *,
    dpi: int = 120,
    highlight_texts: Optional[list[str]] = None,
) -> tuple[Optional[bytes], int]:
    """1-based page_number 페이지를 PNG로 렌더. 하이라이트된 영역 수를 함께 반환."""
    if page_number < 1:
        return None, 0
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None, 0
    try:
        if page_number > len(doc):
            return None, 0
        page = doc[page_number - 1]
        hit_count = 0
        if highlight_texts:
            rects = _find_rects(page, highlight_texts)
            hit_count = _apply_highlights(page, rects)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png"), hit_count
    except Exception:
        return None, 0
    finally:
        doc.close()


def collect_matched_page_pairs(sentence_pairs: list[dict]) -> list[tuple[str, int, str, int]]:
    """
    유사 문장 쌍에서 (파일A, 페이지A, 파일B, 페이지B)를 중복 없이 수집.
    확인이 쉽도록 A/B 페이지가 모두 있는 쌍만 포함한다.
    """
    details = collect_matched_page_pair_details(sentence_pairs)
    return [(d["file_a"], d["page_a"], d["file_b"], d["page_b"]) for d in details]


def collect_matched_page_pair_details(sentence_pairs: list[dict]) -> list[dict]:
    """
    페이지 쌍별로 하이라이트할 문장 텍스트를 모아 반환.

    Returns:
        [{file_a, page_a, file_b, page_b, texts_a, texts_b, pair_count}, ...]
    """
    groups: OrderedDict[tuple[str, int, str, int], dict] = OrderedDict()

    for p in sentence_pairs:
        pa = parse_page_number(p.get("location_a", ""))
        pb = parse_page_number(p.get("location_b", ""))
        if pa is None or pb is None:
            continue
        fa, fb = p["file_a"], p["file_b"]
        ta = p.get("text_a") or ""
        tb = p.get("text_b") or ""

        if (fa, pa) <= (fb, pb):
            key = (fa, pa, fb, pb)
            text_a, text_b = ta, tb
        else:
            key = (fb, pb, fa, pa)
            text_a, text_b = tb, ta

        if key not in groups:
            groups[key] = {
                "file_a": key[0],
                "page_a": key[1],
                "file_b": key[2],
                "page_b": key[3],
                "texts_a": [],
                "texts_b": [],
                "pair_count": 0,
            }
        g = groups[key]
        g["pair_count"] += 1
        if text_a and text_a not in g["texts_a"]:
            g["texts_a"].append(text_a)
        if text_b and text_b not in g["texts_b"]:
            g["texts_b"].append(text_b)

    return list(groups.values())


def _pair_stem(file_a: str, page_a: int, file_b: str, page_b: int) -> str:
    """파일A_p0003=파일B_p0007 형태 공통 파일명 stem."""
    return (
        f"{_safe_name(file_a)}_p{page_a:04d}"
        f"={_safe_name(file_b)}_p{page_b:04d}"
    )


def stitch_side_by_side(
    png_a: bytes,
    png_b: bytes,
    *,
    caption_a: str = "A",
    caption_b: str = "B",
    gap: int = 20,
    header_h: int = 40,
    bg=(255, 255, 255),
) -> Optional[bytes]:
    """미리보기처럼 A|B 페이지를 한 장 PNG로 가로 결합."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    try:
        img_a = Image.open(io.BytesIO(png_a)).convert("RGB")
        img_b = Image.open(io.BytesIO(png_b)).convert("RGB")
    except Exception:
        return None

    # 높이 맞추기 (작은 쪽에 맞춤 후 여백)
    target_h = max(img_a.height, img_b.height)

    def _pad(im: Image.Image) -> Image.Image:
        if im.height == target_h:
            return im
        canvas = Image.new("RGB", (im.width, target_h), bg)
        canvas.paste(im, (0, (target_h - im.height) // 2))
        return canvas

    img_a = _pad(img_a)
    img_b = _pad(img_b)
    width = img_a.width + gap + img_b.width
    height = header_h + target_h
    out = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((8, 10), caption_a[:80], fill=(30, 30, 30), font=font)
    draw.text((img_a.width + gap + 8, 10), caption_b[:80], fill=(30, 30, 30), font=font)
    # 구분선
    x_div = img_a.width + gap // 2
    draw.line([(x_div, header_h), (x_div, height)], fill=(200, 200, 200), width=2)

    out.paste(img_a, (0, header_h))
    out.paste(img_b, (img_a.width + gap, header_h))

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def build_matched_page_screenshots(
    pdf_bytes_by_name: dict[str, bytes],
    page_pairs: list,
    *,
    dpi: int = 120,
) -> list[dict]:
    """
    유사 문장 페이지 쌍을 PNG로 렌더 (매칭 문장 노란색 하이라이트).

    각 쌍에 대해 A/B 개별 PNG와, 나란히 합친 AB PNG를 만든다.
    ZIP 다운로드는 AB(합본)만 사용한다.

    page_pairs: collect_matched_page_pairs 결과 또는
                collect_matched_page_pair_details 결과.
    """
    # tuple 목록이면 detail 없이 하이라이트 없이 렌더 (하위호환)
    if page_pairs and isinstance(page_pairs[0], tuple):
        details = [
            {
                "file_a": fa,
                "page_a": pa,
                "file_b": fb,
                "page_b": pb,
                "texts_a": [],
                "texts_b": [],
                "pair_count": 0,
            }
            for fa, pa, fb, pb in page_pairs
        ]
    else:
        details = list(page_pairs)

    results: list[dict] = []

    for d in details:
        file_a, page_a = d["file_a"], d["page_a"]
        file_b, page_b = d["file_b"], d["page_b"]
        stem = _pair_stem(file_a, page_a, file_b, page_b)
        pair_label = f"{file_a} p.{page_a} = {file_b} p.{page_b}"
        texts_a = d.get("texts_a") or []
        texts_b = d.get("texts_b") or []

        sides: dict[str, dict] = {}
        for side, fname, page, texts in (
            ("A", file_a, page_a, texts_a),
            ("B", file_b, page_b, texts_b),
        ):
            pdf_bytes = pdf_bytes_by_name.get(fname)
            if not pdf_bytes:
                continue
            png, hit_count = render_page_png(
                pdf_bytes, page, dpi=dpi, highlight_texts=texts or None
            )
            if not png:
                continue
            item = {
                "pair_label": pair_label,
                "side": side,
                "file_name": fname,
                "page_number": page,
                "file_a": file_a,
                "page_a": page_a,
                "file_b": file_b,
                "page_b": page_b,
                "filename": f"{stem}__{side}.png",
                "png_bytes": png,
                "highlight_texts": texts,
                "highlight_hits": hit_count,
                "pair_count": d.get("pair_count", 0),
            }
            sides[side] = item
            results.append(item)

        # 나란히 합본 (다운로드용)
        if "A" in sides and "B" in sides:
            combined = stitch_side_by_side(
                sides["A"]["png_bytes"],
                sides["B"]["png_bytes"],
                caption_a=f"A: {file_a}  p.{page_a}",
                caption_b=f"B: {file_b}  p.{page_b}",
            )
            if combined:
                results.append(
                    {
                        "pair_label": pair_label,
                        "side": "AB",
                        "file_name": f"{file_a} | {file_b}",
                        "page_number": page_a,
                        "file_a": file_a,
                        "page_a": page_a,
                        "file_b": file_b,
                        "page_b": page_b,
                        "filename": f"{stem}.png",
                        "png_bytes": combined,
                        "highlight_texts": (texts_a or []) + (texts_b or []),
                        "highlight_hits": sides["A"].get("highlight_hits", 0)
                        + sides["B"].get("highlight_hits", 0),
                        "pair_count": d.get("pair_count", 0),
                        "png_a": sides["A"]["png_bytes"],
                        "png_b": sides["B"]["png_bytes"],
                        "hits_a": sides["A"].get("highlight_hits", 0),
                        "hits_b": sides["B"].get("highlight_hits", 0),
                        "texts_a": texts_a,
                        "texts_b": texts_b,
                    }
                )
    return results


def screenshots_to_zip(screenshots: list[dict], *, combined_only: bool = True) -> bytes:
    """스크린샷 목록을 ZIP으로 묶는다. 기본은 A|B 합본(side=AB)만 넣는다."""
    items = list(screenshots or [])
    if combined_only:
        combined = [s for s in items if s.get("side") == "AB"]
        if combined:
            items = combined
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        used: set[str] = set()
        for item in items:
            name = item["filename"]
            if name in used:
                continue
            used.add(name)
            zf.writestr(name, item["png_bytes"])
    return buf.getvalue()
