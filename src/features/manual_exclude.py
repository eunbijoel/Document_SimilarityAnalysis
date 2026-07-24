"""수동 제외 기능 — 사용자가 입력한 단어/문장으로 유사 문장 결과를 후처리 제외.

이 모듈은 Streamlit UI와 분리된 순수 로직입니다.
UI는 `src/ui/excluded_sentences_tab.py` 를 참고하세요.
"""

from __future__ import annotations

from typing import Any, Optional

from src.utils.result_word_filter import filter_pairs_by_words
from src.utils.summary_stats import (
    compute_sentence_overlap_stats,
    compute_similarity_distribution,
    compute_file_pair_matrix,
)
from src.utils.page_screenshots import (
    build_matched_page_screenshots,
    collect_matched_page_pair_details,
)

# session_state 키 (UI·app 공통)
SESSION_TERMS = "manual_exclude_terms"
SESSION_VIEW = "manual_exclude_view"
SESSION_TEXT = "manual_exclude_text"


def parse_manual_exclude_terms(raw: str) -> list[str]:
    """줄바꿈·쉼표로 구분된 제외 문구를 파싱."""
    terms: list[str] = []
    seen: set[str] = set()
    for line in (raw or "").replace(",", "\n").splitlines():
        w = line.strip()
        if not w or w in seen:
            continue
        seen.add(w)
        terms.append(w)
    return terms


def build_manual_exclude_view(analysis: dict, terms: list[str]) -> dict[str, Any]:
    """
    수동 입력 문구가 포함된 유사 문장 쌍을 결과·통계·PNG에서 제외한 뷰를 만든다.

    Returns:
        analysis에 merge할 수 있는 필드 dict
        (sentence_pairs, overlap_stats, file_matrix, matched_page_pngs, ...)
    """
    base_pairs = analysis.get("sentence_pairs_base") or analysis.get("sentence_pairs") or []
    kept, removed = filter_pairs_by_words(base_pairs, terms)
    file_names = analysis.get("file_names") or []
    overlap = compute_sentence_overlap_stats(analysis.get("sentences") or [], kept)
    sim_dist = compute_similarity_distribution(kept)
    file_matrix = compute_file_pair_matrix(kept, file_names)
    pdf_bytes = analysis.get("pdf_bytes_by_name") or {}
    details = collect_matched_page_pair_details(kept)
    matched_pngs = (
        build_matched_page_screenshots(pdf_bytes, details, dpi=120) if pdf_bytes else []
    )
    return {
        "sentence_pairs": kept,
        "overlap_stats": overlap,
        "similarity_distribution": sim_dist,
        "file_matrix": file_matrix,
        "matched_page_pngs": matched_pngs,
        "manual_excluded_pairs": removed,
        "manual_exclude_terms": list(terms),
    }


def merge_display_analysis(analysis: dict, manual_view: Optional[dict] = None) -> dict:
    """수동 제외 뷰가 있으면 analysis 위에 덮어쓴 표시용 dict를 반환."""
    if not manual_view:
        return analysis
    merged = dict(analysis)
    merged.update(manual_view)
    return merged


def clear_manual_exclude_session_keys(session: dict) -> None:
    """session_state 유사 dict에서 수동 제외 관련 키를 제거."""
    for key in (SESSION_TERMS, SESSION_VIEW, SESSION_TEXT):
        session.pop(key, None)
