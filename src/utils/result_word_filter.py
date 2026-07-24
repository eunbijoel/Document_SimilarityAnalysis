"""유사 문장 결과에서 사용자가 고른 단어로 쌍을 후처리 제외."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

# 결과에서 자주 쓰일 법한 양식·기관 후보 (쌍에 실제로 있을 때만 체크박스에 표시)
DEFAULT_SUGGESTED_TERMS = (
    "한국전자기술연구원",
    "기관명",
    "기술 요약 정보",
    "매출 실적",
    "주관연구개발기관",
    "공동연구개발기관",
    "총괄책임자",
    "참여연구원",
    "연구개발과제의 개요",
    "비밀유지",
    "보안등급",
    "문서번호",
    "작성일자",
    "제출일",
    "양식",
    "서식",
    "(단위",
    "단위:",
    "요약문",
)

_TOKEN_RE = re.compile(r"[가-힣]{2,12}|[A-Za-z][A-Za-z0-9\-]{1,20}")


def pair_contains_any(pair: dict, words: Iterable[str]) -> list[str]:
    """쌍의 문장에 포함된 제외 단어 목록을 반환 (없으면 빈 리스트)."""
    if not words:
        return []
    blob = f"{pair.get('text_a') or ''}\n{pair.get('text_b') or ''}"
    hits = []
    for w in words:
        w = (w or "").strip()
        if w and w in blob:
            hits.append(w)
    return hits


def filter_pairs_by_words(
    sentence_pairs: list[dict],
    words: Iterable[str],
) -> tuple[list[dict], list[dict]]:
    """
    선택한 단어가 text_a/text_b에 하나라도 있으면 제외.

    Returns:
        (남은 쌍, 제외된 쌍 — excluded_by 필드 포함)
    """
    words = [w.strip() for w in words if (w or "").strip()]
    if not words:
        return list(sentence_pairs), []

    kept: list[dict] = []
    removed: list[dict] = []
    for p in sentence_pairs:
        hits = pair_contains_any(p, words)
        if hits:
            row = dict(p)
            row["excluded_by"] = hits
            removed.append(row)
        else:
            kept.append(p)
    return kept, removed


def suggest_terms_from_pairs(
    sentence_pairs: list[dict],
    *,
    known_terms: Iterable[str] = DEFAULT_SUGGESTED_TERMS,
    top_n: int = 36,
    min_token_count: int = 2,
) -> list[str]:
    """
    체크박스용 후보 단어.

    1) 사전 양식 용어 중 실제 유사 쌍에 등장하는 것
    2) 유사 쌍에서 자주 나온 짧은 토큰 (빈도순)
    """
    if not sentence_pairs:
        return []

    blobs = [
        f"{p.get('text_a') or ''}\n{p.get('text_b') or ''}" for p in sentence_pairs
    ]
    joined = "\n".join(blobs)

    ordered: list[str] = []
    seen: set[str] = set()

    for term in known_terms:
        t = term.strip()
        if t and t in joined and t not in seen:
            seen.add(t)
            ordered.append(t)

    counter: Counter[str] = Counter()
    for blob in blobs:
        for tok in _TOKEN_RE.findall(blob):
            if len(tok) < 2:
                continue
            # 숫자만/너무 흔한 조사성 짧은 조각은 스킵하지 않되, 과다 후보는 빈도·길이로 정리
            counter[tok] += 1

    for tok, cnt in counter.most_common(200):
        if cnt < min_token_count:
            continue
        if tok in seen:
            continue
        # 이미 known에 가까운 긴 기술 문장 조각은 후보에 넣되 상한 유지
        seen.add(tok)
        ordered.append(tok)
        if len(ordered) >= top_n + len(list(known_terms)):
            break

    # known 우선 + 빈도 토큰을 합친 뒤 top 범위로 자르지 않고,
    # known 전부 + 추가 토큰 top_n
    known_count = sum(1 for t in known_terms if (t or "").strip() in seen)
    extras = [t for t in ordered if t not in set(known_terms)]
    known_present = [t for t in known_terms if (t or "").strip() in seen]
    return known_present + extras[:top_n]


def merge_custom_terms(candidates: list[str], custom: Iterable[str]) -> list[str]:
    """사용자 추가 단어를 후보 앞에 합친다."""
    out: list[str] = []
    seen: set[str] = set()
    for w in list(custom) + list(candidates):
        w = (w or "").strip()
        if not w or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out
