"""제외 문장 탭 UI — 수동 제외 + 자동 제외 복원.

의존:
  - src/features/manual_exclude.py
  - src/features/excluded_restore.py
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st

from src.features import manual_exclude as me
from src.features import excluded_restore as er
from src.models.schemas import SentenceRecord
from src.utils.page_screenshots import (
    build_matched_page_screenshots,
    screenshots_to_zip,
    collect_matched_page_pair_details,
)
from src.utils.result_export import to_csv_bytes


def reset_excluded_feature_session() -> None:
    """분석 재실행 시 제외/복원 관련 session 키를 초기화."""
    for key in (
        "matched_pages_zip",
        "restore_selected_texts",
        er.SESSION_RESTORE_APPLIED,
        me.SESSION_TERMS,
        me.SESSION_VIEW,
        me.SESSION_TEXT,
    ):
        st.session_state.pop(key, None)


def merge_display_analysis(analysis: dict) -> dict:
    """수동 제외가 적용된 표시용 analysis."""
    return me.merge_display_analysis(
        analysis, st.session_state.get(me.SESSION_VIEW)
    )


def render_excluded_tab(
    analysis: dict,
    *,
    recompute_sentence_side: Callable[[dict, list[SentenceRecord], dict], dict],
) -> None:
    """
    제외 문장 탭 전체 렌더.

    Args:
        analysis: 세션에 저장된 원본 분석 결과
        recompute_sentence_side: 문장 복원 시 유사도·PNG를 다시 계산하는 콜백
    """
    rows = analysis.get("excluded_sentences") or []
    stats = analysis.get("boilerplate_stats") or {}
    multi = er.filter_meaningful_excluded(rows, min_file_count=2)

    _render_manual_exclude_section(analysis)
    st.divider()
    _render_restore_section(analysis, multi, stats, rows, recompute_sentence_side)


def _render_manual_exclude_section(analysis: dict) -> None:
    st.subheader("수동 제외 (단어·문장 입력)")
    st.markdown(
        '<p style="color:#c62828;font-weight:700;font-size:0.95rem;margin:0.25rem 0 0.75rem 0;">'
        "한 줄에 하나(또는 쉼표로 구분). "
        "아래 「입력 문구로 결과 제외」를 누를 때만 결과에서 빠집니다."
        "</p>",
        unsafe_allow_html=True,
    )
    with st.form("manual_exclude_form", clear_on_submit=False):
        raw_text = st.text_area(
            "제외할 단어 또는 문장",
            height=100,
            placeholder="예:\n주관연구개발기관\n공동연구개발기관",
            key=me.SESSION_TEXT,
        )
        c1, c2, _ = st.columns([1, 1, 2])
        apply_manual = c1.form_submit_button("입력 문구로 결과 제외", type="primary")
        clear_manual = c2.form_submit_button("수동 제외 해제")

    if clear_manual:
        me.clear_manual_exclude_session_keys(st.session_state)
        st.session_state[me.SESSION_TEXT] = ""
        base_pairs = (
            analysis.get("sentence_pairs_base") or analysis.get("sentence_pairs") or []
        )
        details = collect_matched_page_pair_details(base_pairs)
        pdf_bytes = analysis.get("pdf_bytes_by_name") or {}
        shots = analysis.get("matched_page_pngs") or []
        if not shots and pdf_bytes and base_pairs:
            with st.spinner("페이지 PNG 복원 중..."):
                shots = build_matched_page_screenshots(pdf_bytes, details, dpi=120)
                analysis_u = dict(analysis)
                analysis_u["matched_page_pngs"] = shots
                analysis_u["sentence_pairs"] = base_pairs
                st.session_state["analysis"] = analysis_u
        st.session_state["matched_pages_zip"] = (
            screenshots_to_zip(shots) if shots else None
        )
        st.rerun()

    if apply_manual:
        terms = me.parse_manual_exclude_terms(raw_text)
        if not terms:
            st.warning("제외할 단어/문장을 입력하세요.")
        else:
            with st.spinner(
                "입력 문구로 유사 문장 결과·통계·페이지 PNG를 다시 구성하는 중..."
            ):
                view = me.build_manual_exclude_view(analysis, terms)
            st.session_state[me.SESSION_TERMS] = terms
            st.session_state[me.SESSION_VIEW] = view
            shots = view.get("matched_page_pngs") or []
            st.session_state["matched_pages_zip"] = (
                screenshots_to_zip(shots) if shots else None
            )
            st.rerun()

    terms_applied = st.session_state.get(me.SESSION_TERMS) or []
    view = st.session_state.get(me.SESSION_VIEW)
    if view is not None and terms_applied:
        base_n = len(
            analysis.get("sentence_pairs_base") or analysis.get("sentence_pairs") or []
        )
        kept_n = len(view.get("sentence_pairs") or [])
        rem = view.get("manual_excluded_pairs") or []
        st.success(
            f"수동 제외 적용: 유사 문장 쌍 {base_n} → {kept_n} "
            f"(제외 {len(rem)}쌍 · 문구 {len(terms_applied)}개)"
        )
        if rem:
            with st.expander(f"수동 제외된 유사 문장 쌍 {len(rem)}개", expanded=False):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "제외 문구": ", ".join(r.get("excluded_by") or []),
                                "파일 A": r.get("file_a"),
                                "파일 B": r.get("file_b"),
                                "문장 A": r.get("text_a"),
                                "문장 B": r.get("text_b"),
                                "유사도": r.get("similarity"),
                            }
                            for r in rem
                        ]
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


def _render_restore_section(
    analysis: dict,
    multi: list[dict],
    stats: dict,
    rows: list[dict],
    recompute_sentence_side: Callable,
) -> None:
    st.subheader("자동 제외 문장")

    if not rows:
        st.info("자동 제외된 문장이 없습니다. (필터가 꺼져 있거나, 해당 문장이 없었습니다.)")
        return

    st.caption(
        f"자동 제외 전체 {len(rows)}개 중 유의미 중복(등장 파일 수≥2) {len(multi)}개만 표시 "
        f"(패턴 {stats.get('removed_pattern', 0)} · "
        f"라벨형 {stats.get('removed_label', 0)} · "
        f"파일공통 {stats.get('removed_common', 0)}). "
        "등장 1회 문장은 표시하지 않습니다."
    )
    st.info(
        "목록에서 고른 뒤 「다시 포함」을 누를 때만 재계산합니다. "
        "(항목을 고르는 동안에는 페이지가 다시 돌지 않습니다.)"
    )

    groups = er.group_excluded_by_text(multi)
    if not groups:
        st.warning("표시할 유의미 중복 제외 문장이 없습니다.")
    else:
        reason_opts = sorted({g["reason"] for g in groups.values() if g["reason"]})
        reason_filter = st.multiselect(
            "제외 사유 필터",
            options=reason_opts,
            default=reason_opts,
            key="excluded_reason_filter",
        )
        labels, label_to_key = er.build_restore_labels(
            groups, reason_filter=reason_filter or None
        )

        with st.form("restore_excluded_form"):
            st.caption(
                f"선택 가능 {len(labels)}개 · 검색어로 좁힌 뒤 여러 개 선택하고 "
                "「다시 포함」을 누르세요. (선택 중에는 재실행되지 않습니다)"
            )
            selected_labels = st.multiselect(
                "결과에 다시 포함할 문장",
                options=labels,
                default=[],
            )
            submitted = st.form_submit_button(
                "선택 문장 결과에 다시 포함", type="primary"
            )

        if submitted:
            checked_keys = [
                label_to_key[lb] for lb in selected_labels if lb in label_to_key
            ]
            if not checked_keys:
                st.warning("다시 포함할 문장을 하나 이상 선택하세요.")
            else:
                settings = st.session_state.get("analysis_settings") or {}
                with st.spinner("선택 문장을 포함해 유사 문장·통계·페이지 PNG를 다시 계산 중..."):
                    analysis_new, extra = er.apply_restore(
                        analysis,
                        checked_keys,
                        groups,
                        settings=settings,
                        recompute_fn=recompute_sentence_side,
                    )
                st.session_state["analysis"] = analysis_new
                st.session_state[er.SESSION_RESTORE_APPLIED] = checked_keys
                st.session_state.pop(me.SESSION_VIEW, None)
                st.session_state.pop(me.SESSION_TERMS, None)
                shots = analysis_new.get("matched_page_pngs") or []
                st.session_state["matched_pages_zip"] = (
                    screenshots_to_zip(shots) if shots else None
                )
                st.success(
                    f"다시 포함 완료: 문장 +{len(extra)} · "
                    f"유사 문장 쌍 {len(analysis_new.get('sentence_pairs') or [])}개"
                )
                st.rerun()

    if multi:
        st.markdown("#### 유의미 중복 제외 목록 (표)")
        df = pd.DataFrame(multi)
        rename = {
            "file_name": "파일명",
            "location": "위치",
            "text": "제외 문장",
            "reason": "제외 사유",
            "file_count": "등장 파일 수",
        }
        df = df.rename(columns=rename)
        cols = [
            c
            for c in ["파일명", "위치", "제외 사유", "등장 파일 수", "제외 문장"]
            if c in df.columns
        ]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        st.download_button(
            "excluded_sentences_meaningful.csv 다운로드",
            data=to_csv_bytes(df[cols]),
            file_name="excluded_sentences_meaningful.csv",
            mime="text/csv",
            key="dl_excluded_meaningful",
        )
