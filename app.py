"""문서 간 유사 콘텐츠 분석기

여러 PDF 문서를 업로드하면 동일하거나 유사한 문장/페이지/이미지를 찾아 보여주는
내부 검토용 Streamlit 도구입니다. 표절 여부를 자동으로 판정하지 않습니다.
"""
import os

# Keras 3 / Transformers 충돌 방지 (다른 import보다 먼저)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import traceback

import streamlit as st

from src.parsers.pdf_parser import parse_pdf
from src.analyzers.text_similarity import (
    load_model,
    find_exact_duplicate_pairs,
    find_similar_sentence_pairs,
    find_similar_page_pairs,
)
from src.analyzers.image_similarity import compute_phash, find_similar_image_pairs
from src.utils.result_export import (
    sentence_pairs_to_df,
    page_pairs_to_df,
    image_pairs_to_df,
    to_csv_bytes,
    build_excel_report,
)
from src.utils.config import (
    SENTENCE_MODEL_NAME,
    DEFAULT_SENTENCE_THRESHOLD,
    DEFAULT_PAGE_THRESHOLD,
    DEFAULT_MIN_SENTENCE_LENGTH,
    DEFAULT_MIN_IMAGE_WIDTH,
    DEFAULT_MIN_IMAGE_HEIGHT,
    DEFAULT_PHASH_DISTANCE_THRESHOLD,
    DEFAULT_MAX_RESULTS,
    DEFAULT_MAX_SENTENCES,
)
from src.utils.summary_stats import (
    compute_sentence_overlap_stats,
    compute_similarity_distribution,
    compute_file_pair_matrix,
)
from src.utils.boilerplate_filter import filter_boilerplate_sentences
from src.utils.result_word_filter import filter_pairs_by_words
from src.utils.page_screenshots import (
    build_matched_page_screenshots,
    screenshots_to_zip,
    collect_matched_page_pair_details,
)
from src.models.schemas import SentenceRecord

st.set_page_config(page_title="문서 간 유사 콘텐츠 분석기", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(SENTENCE_MODEL_NAME)


def friendly_error(exc: Exception) -> str:
    """개발자용 traceback 대신 사용자에게 보여줄 간단한 오류 메시지를 만듭니다."""
    return f"{type(exc).__name__}: {exc}"


def _reset_result_filter_state():
    for key in (
        "matched_pages_zip",
        "restore_selected_texts",
        "excluded_restore_applied",
        "manual_exclude_terms",
        "manual_exclude_view",
        "manual_exclude_text",
    ):
        st.session_state.pop(key, None)


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


def build_manual_exclude_view(analysis: dict, terms: list[str]) -> dict:
    """수동 입력 문구가 포함된 유사 문장 쌍을 결과·통계·PNG에서 제외."""
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


def merge_display_analysis(analysis: dict) -> dict:
    view = st.session_state.get("manual_exclude_view")
    if not view:
        return analysis
    merged = dict(analysis)
    merged.update(view)
    return merged


def _row_to_sentence(row: dict) -> SentenceRecord:
    text = row.get("text") or ""
    norm = (row.get("normalized_text") or text).strip()
    return SentenceRecord(
        file_name=row.get("file_name") or "",
        file_type=row.get("file_type") or "pdf",
        location=row.get("location") or "",
        text=text,
        normalized_text=norm,
        sentence_id=row.get("sentence_id") or "",
    )


def recompute_sentence_side(analysis: dict, sentences: list[SentenceRecord], settings: dict) -> dict:
    """문장 목록으로 유사 문장·통계·페이지 PNG만 다시 계산한다."""
    sentence_pairs: list[dict] = []
    model = None
    try:
        model = get_model()
    except Exception as exc:  # noqa: BLE001
        st.error(f"문장 임베딩 모델 로딩 중 오류: {friendly_error(exc)}")

    if model is not None and len(sentences) >= 2:
        exact_pairs, exact_keys = find_exact_duplicate_pairs(
            sentences, include_same_file=settings.get("include_same_file", False)
        )
        sentence_pairs.extend(exact_pairs)
        try:
            similar_pairs = find_similar_sentence_pairs(
                sentences,
                model,
                threshold=settings.get("sentence_threshold", DEFAULT_SENTENCE_THRESHOLD),
                include_same_file=settings.get("include_same_file", False),
                already_found=exact_keys,
            )
            sentence_pairs.extend(similar_pairs)
        except Exception as exc:  # noqa: BLE001
            st.error(f"문장 유사도 분석 오류: {friendly_error(exc)}")

    sentence_pairs.sort(key=lambda p: p["similarity"], reverse=True)
    max_results = int(settings.get("max_results", DEFAULT_MAX_RESULTS))
    sentence_pairs = sentence_pairs[:max_results]

    file_names = analysis.get("file_names") or []
    overlap = compute_sentence_overlap_stats(sentences, sentence_pairs)
    sim_dist = compute_similarity_distribution(sentence_pairs)
    file_matrix = compute_file_pair_matrix(sentence_pairs, file_names)
    pdf_bytes = analysis.get("pdf_bytes_by_name") or {}
    details = collect_matched_page_pair_details(sentence_pairs)
    matched_pngs = (
        build_matched_page_screenshots(pdf_bytes, details, dpi=120) if pdf_bytes else []
    )
    return {
        "sentences": sentences,
        "sentence_pairs": sentence_pairs,
        "sentence_pairs_base": sentence_pairs,
        "overlap_stats": overlap,
        "similarity_distribution": sim_dist,
        "file_matrix": file_matrix,
        "matched_page_pngs": matched_pngs,
    }


def run_analysis(uploaded_files, settings):
    log_entries = []
    all_sentences = []
    all_images = []
    all_pages = []
    pdf_bytes_by_name: dict[str, bytes] = {}

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    total = len(uploaded_files)
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"처리 중: {uploaded_file.name} ({idx + 1}/{total})")
        try:
            file_bytes = uploaded_file.read()
            pdf_bytes_by_name[uploaded_file.name] = file_bytes
            result = parse_pdf(
                uploaded_file.name,
                file_bytes,
                min_sentence_length=settings["min_sentence_length"],
                min_image_width=settings["min_image_width"],
                min_image_height=settings["min_image_height"],
            )
            if result.success:
                all_sentences.extend(result.sentences)
                all_images.extend(result.images)
                all_pages.extend(result.pages)
                log_entries.append(
                    {
                        "file_name": uploaded_file.name,
                        "status": "성공",
                        "message": (
                            f"페이지 {len(result.pages)}개, 문장 {len(result.sentences)}개, "
                            f"이미지 {len(result.images)}개 추출"
                        ),
                    }
                )
            else:
                log_entries.append(
                    {"file_name": uploaded_file.name, "status": "실패", "message": result.error_message}
                )
        except Exception as exc:  # noqa: BLE001
            log_entries.append(
                {"file_name": uploaded_file.name, "status": "실패", "message": friendly_error(exc)}
            )
            st.session_state.setdefault("_tracebacks", []).append(traceback.format_exc())
        progress_bar.progress((idx + 1) / total)

    # --- 양식/공통 문장 제외 (유사 비교 전에 적용) ---
    n_files = len({f.name for f in uploaded_files})
    boilerplate_stats = {
        "input": len(all_sentences),
        "removed_total": 0,
        "kept": len(all_sentences),
        "removed_pattern": 0,
        "removed_label": 0,
        "removed_common": 0,
        "common_threshold": 0,
        "n_files": n_files,
    }
    excluded_sentences: list[dict] = []
    if settings.get("exclude_boilerplate_patterns") or settings.get("exclude_common_sentences"):
        status_text.text("양식/공통 문장 필터링 중...")
        all_sentences, boilerplate_stats, excluded_sentences = filter_boilerplate_sentences(
            all_sentences,
            n_files=n_files,
            use_patterns=bool(settings.get("exclude_boilerplate_patterns", True)),
            use_common_across_files=bool(settings.get("exclude_common_sentences", True)),
            common_max_length=int(settings.get("common_max_length", 20)),
        )
        removed = boilerplate_stats.get("removed_total", 0)
        if removed:
            st.info(
                f"양식/공통 문장 {removed}개를 비교에서 제외했습니다 "
                f"(패턴 {boilerplate_stats.get('removed_pattern', 0)} · "
                f"라벨형 {boilerplate_stats.get('removed_label', 0)} · "
                f"파일공통 {boilerplate_stats.get('removed_common', 0)}, "
                f"기준≥{boilerplate_stats.get('common_threshold')}개 파일). "
                f"남은 문장 {boilerplate_stats.get('kept', 0)}개. "
                "자세한 목록은「제외 문장」탭에서 확인할 수 있습니다."
            )

    # --- 문장 hard-cap (대용량 PDF 안전장치, 초안 max_sentences=5000) ---
    max_sentences = settings["max_sentences"]
    truncated = False
    if len(all_sentences) > max_sentences:
        truncated = True
        st.warning(
            f"추출 문장(필터 후) {len(all_sentences)}개 → 비교는 상위 {max_sentences}개만 사용합니다. "
            "(대용량 PDF 실행시간·메모리 보호용 hard-cap)"
        )
        all_sentences = all_sentences[:max_sentences]
    elif len(all_sentences) > max_sentences * 0.8:
        st.info(f"비교 대상 문장 {len(all_sentences)}개 (상한 {max_sentences}).")

    status_text.text("페이지·문장 유사도 분석 중입니다...")

    page_pairs = []
    sentence_pairs = []
    model = None

    try:
        model = get_model()
    except Exception as exc:  # noqa: BLE001
        st.error(f"문장 임베딩 모델 로딩 중 오류: {friendly_error(exc)}")

    if model is not None and len(all_pages) >= 2:
        status_text.text("유사 페이지 분석 중...")
        try:
            page_pairs = find_similar_page_pairs(
                all_pages,
                model,
                threshold=settings["page_threshold"],
                include_same_file=settings["include_same_file"],
            )
            page_pairs = page_pairs[: settings["max_results"]]
        except Exception as exc:  # noqa: BLE001
            st.error(f"페이지 유사도 분석 오류: {friendly_error(exc)}")

    if model is not None and len(all_sentences) >= 2:
        status_text.text("유사 문장 분석 중...")
        exact_pairs, exact_keys = find_exact_duplicate_pairs(
            all_sentences, include_same_file=settings["include_same_file"]
        )
        sentence_pairs.extend(exact_pairs)
        try:
            similar_pairs = find_similar_sentence_pairs(
                all_sentences,
                model,
                threshold=settings["sentence_threshold"],
                include_same_file=settings["include_same_file"],
                already_found=exact_keys,
            )
            sentence_pairs.extend(similar_pairs)
        except Exception as exc:  # noqa: BLE001
            st.error(f"문장 유사도 분석 오류: {friendly_error(exc)}")

    sentence_pairs.sort(key=lambda p: p["similarity"], reverse=True)
    sentence_pairs = sentence_pairs[: settings["max_results"]]

    # --- 이미지 유사도 ---
    image_pairs = []
    if settings["enable_image_analysis"] and len(all_images) >= 2:
        status_text.text("이미지 유사도 분석 중입니다...")
        for img in all_images:
            compute_phash(img)
        image_pairs = find_similar_image_pairs(
            all_images,
            distance_threshold=settings["phash_threshold"],
            include_same_file=settings["include_same_file"],
        )
        image_pairs.sort(key=lambda p: p["phash_distance"])
        image_pairs = image_pairs[: settings["max_results"]]

    # --- 유사 문장 페이지 PNG 렌더 (매칭 문장 하이라이트) ---
    status_text.text("유사 문장 페이지 스크린샷 생성 중...")
    page_pairs_for_png = collect_matched_page_pair_details(sentence_pairs)
    matched_page_pngs = build_matched_page_screenshots(
        pdf_bytes_by_name, page_pairs_for_png, dpi=120
    )

    status_text.text("분석이 완료되었습니다.")
    progress_bar.progress(1.0)

    file_names = [f.name for f in uploaded_files]
    overlap_stats = compute_sentence_overlap_stats(all_sentences, sentence_pairs)
    sim_dist = compute_similarity_distribution(sentence_pairs)
    file_matrix = compute_file_pair_matrix(sentence_pairs, file_names)

    return {
        "sentences": all_sentences,
        "images": all_images,
        "pages": all_pages,
        "sentence_pairs": sentence_pairs,
        "sentence_pairs_base": list(sentence_pairs),
        "page_pairs": page_pairs,
        "image_pairs": image_pairs,
        "log_entries": log_entries,
        "sentences_truncated": truncated,
        "pdf_bytes_by_name": pdf_bytes_by_name,
        "matched_page_pngs": matched_page_pngs,
        "overlap_stats": overlap_stats,
        "similarity_distribution": sim_dist,
        "file_matrix": file_matrix,
        "file_names": file_names,
        "boilerplate_stats": boilerplate_stats,
        "excluded_sentences": excluded_sentences,
    }


def _thin_bar_chart(df, x_col: str, y_col: str, *, x_sort=None, height: int = 280):
    """범주형 X축용 얇은 막대 그래프."""
    import altair as alt

    x_enc = alt.X(
        f"{x_col}:N",
        sort=x_sort,
        axis=alt.Axis(labelAngle=0, labelLimit=160, title=x_col),
        scale=alt.Scale(paddingInner=0.65, paddingOuter=0.25),
    )
    y_enc = alt.Y(f"{y_col}:Q", axis=alt.Axis(title=y_col))
    chart = (
        alt.Chart(df)
        .mark_bar(size=18, cornerRadiusEnd=2)
        .encode(x=x_enc, y=y_enc, tooltip=[x_col, y_col])
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridOpacity=0.3)
    )
    st.altair_chart(chart, use_container_width=True)


def _similarity_hist_chart(df, *, height: int = 320):
    """유사도 0.01 단위 막대 그래프 — 값이 있는 구간만, 막대 위 개수 표시."""
    import altair as alt

    if df is None or df.empty:
        return

    plot_df = df[df["쌍 개수"] > 0].copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("유사도").reset_index(drop=True)
    plot_df["유사도_라벨"] = plot_df["유사도"].map(lambda x: f"{float(x):.2f}")
    label_order = plot_df["유사도_라벨"].tolist()

    # 막대 너비: 구간 수에 따라 조절 (너무 많으면 조금 얇게)
    n = len(plot_df)
    bar_size = 36 if n <= 12 else (28 if n <= 20 else 18)

    bars = (
        alt.Chart(plot_df)
        .mark_bar(size=bar_size, cornerRadiusEnd=3, color="#4C78A8")
        .encode(
            x=alt.X(
                "유사도_라벨:N",
                sort=label_order,
                axis=alt.Axis(title="유사도", labelAngle=0, labelFontSize=12),
            ),
            y=alt.Y(
                "쌍 개수:Q",
                axis=alt.Axis(title="쌍 개수"),
                scale=alt.Scale(domainMin=0),
            ),
            tooltip=[
                alt.Tooltip("유사도:Q", format=".2f", title="유사도"),
                alt.Tooltip("쌍 개수:Q", title="쌍 개수"),
            ],
        )
    )
    labels = (
        alt.Chart(plot_df)
        .mark_text(dy=-10, fontSize=12, fontWeight="bold", color="#222")
        .encode(
            x=alt.X("유사도_라벨:N", sort=label_order),
            y=alt.Y("쌍 개수:Q"),
            text=alt.Text("쌍 개수:Q"),
        )
    )
    chart = (
        (bars + labels)
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridOpacity=0.25)
    )
    st.altair_chart(chart, use_container_width=True)


def render_summary_tab(analysis, file_count: int | None = None):
    import pandas as pd

    log_entries = analysis["log_entries"]
    success_count = sum(1 for e in log_entries if e["status"] == "성공")
    fail_count = sum(1 for e in log_entries if e["status"] == "실패")
    overlap = analysis.get("overlap_stats") or compute_sentence_overlap_stats(
        analysis["sentences"], analysis["sentence_pairs"]
    )
    names = analysis.get("file_names") or []
    n_files = file_count if file_count is not None else len(names)

    st.subheader("처리 로그")
    if log_entries:
        st.dataframe(pd.DataFrame(log_entries), use_container_width=True, hide_index=True)
    else:
        st.info("처리 로그가 없습니다.")

    st.subheader("기본 지표")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("파일 수", n_files)
    col1.metric("정상 처리", success_count)
    col2.metric("실패", fail_count)
    col2.metric("추출 페이지", len(analysis.get("pages", [])))
    col3.metric("추출 문장", overlap["total_sentences"])
    col3.metric("유사 문장 쌍", len(analysis["sentence_pairs"]))
    col4.metric("유사 페이지 쌍", len(analysis.get("page_pairs", [])))
    col4.metric("유사 이미지 쌍", len(analysis["image_pairs"]))

    bp = analysis.get("boilerplate_stats") or {}
    if bp.get("removed_total"):
        st.caption(
            f"양식/공통 필터로 제외한 문장 {bp['removed_total']}개 "
            f"(패턴 {bp.get('removed_pattern', 0)} · 라벨형 {bp.get('removed_label', 0)} · "
            f"파일공통 {bp.get('removed_common', 0)}, "
            f"공통 기준 ≥{bp.get('common_threshold')}개 파일 / 전체 {bp.get('n_files')}개)."
        )

    # --- 1) 전체 문장 대비 겹침 ---
    st.subheader("문장 겹침 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("추출 문장 수", overlap["total_sentences"])
    c2.metric("겹친 문장 수", overlap["overlapped_sentences"])
    c3.metric("겹침 비율", f"{overlap['overlap_ratio_pct']}%")
    st.caption("겹친 문장 = 유사 문장 쌍에 한 번이라도 등장한 문장(중복 제거).")
    per_file_df = overlap.get("per_file_df")
    if per_file_df is not None and not per_file_df.empty:
        st.markdown("**파일별 문장 겹침**")
        st.dataframe(per_file_df, use_container_width=True, hide_index=True)

    # --- 2) 유사도 구간 분포 그래프 (0.01 단위) ---
    st.subheader("유사도 구간 분포 (문장 쌍)")
    dist_df = compute_similarity_distribution(analysis["sentence_pairs"])
    if dist_df is not None and not dist_df.empty and (dist_df["쌍 개수"] > 0).any():
        _similarity_hist_chart(dist_df)
        st.caption("막대 위 숫자 = 해당 유사도(0.01 단위)의 문장 쌍 개수")
    else:
        st.info("유사 문장 쌍이 없어 분포를 그릴 수 없습니다.")

    # --- 3) 파일×파일 매트릭스 ---
    st.subheader("파일 × 파일 유사 문장 쌍 수")
    matrix = analysis.get("file_matrix")
    if matrix is None or (hasattr(matrix, "empty") and matrix.empty):
        names = analysis.get("file_names") or []
        matrix = compute_file_pair_matrix(analysis["sentence_pairs"], names)
    if matrix is not None and not matrix.empty:
        st.dataframe(matrix, use_container_width=True)
        st.caption("행·열 = 파일명, 칸 값 = 두 파일 사이 유사 문장 쌍 개수 (파일이 늘면 행/열이 늘어납니다).")
        participation = matrix.sum(axis=1).astype(int)
        chart_df = pd.DataFrame(
            {"파일": participation.index.astype(str), "유사쌍_참여수": participation.values}
        )
        _thin_bar_chart(chart_df, "파일", "유사쌍_참여수")
        st.caption("파일별 유사 문장 쌍 참여 횟수 합계")
    else:
        st.info("파일 매트릭스를 만들 데이터가 없습니다.")

    st.subheader("파일별 발견 건수")
    per_file_counts = {}
    for pair in analysis.get("page_pairs", []):
        for f in (pair["file_a"], pair["file_b"]):
            per_file_counts.setdefault(f, {"유사 페이지": 0, "유사 문장": 0, "유사 이미지": 0})
            per_file_counts[f]["유사 페이지"] += 1
    for pair in analysis["sentence_pairs"]:
        for f in (pair["file_a"], pair["file_b"]):
            per_file_counts.setdefault(f, {"유사 페이지": 0, "유사 문장": 0, "유사 이미지": 0})
            per_file_counts[f]["유사 문장"] += 1
    for pair in analysis["image_pairs"]:
        for f in (pair["file_a"], pair["file_b"]):
            per_file_counts.setdefault(f, {"유사 페이지": 0, "유사 문장": 0, "유사 이미지": 0})
            per_file_counts[f]["유사 이미지"] += 1

    if per_file_counts:
        df = pd.DataFrame.from_dict(per_file_counts, orient="index")
        df.index.name = "파일명"
        st.dataframe(df, use_container_width=True)
    else:
        st.info("발견된 유사 항목이 없습니다.")

    return {
        "파일 수": n_files,
        "정상 처리 파일 수": success_count,
        "실패 파일 수": fail_count,
        "추출 페이지 수": len(analysis.get("pages", [])),
        "추출 문장 수": overlap["total_sentences"],
        "겹친 문장 수": overlap["overlapped_sentences"],
        "문장 겹침 비율(%)": overlap["overlap_ratio_pct"],
        "추출 이미지 수": len(analysis["images"]),
        "유사 페이지 쌍 수": len(analysis.get("page_pairs", [])),
        "유사 문장 쌍 수": len(analysis["sentence_pairs"]),
        "유사 이미지 쌍 수": len(analysis["image_pairs"]),
        "매칭 페이지 PNG 수": len(analysis.get("matched_page_pngs") or []),
    }


def render_matched_pages_tab(analysis):
    """유사 문장이 포함된 페이지 PNG — 나란히 합본으로 미리보기·ZIP 다운로드."""
    shots = analysis.get("matched_page_pngs") or []
    if not shots:
        st.info("유사 문장 쌍에서 추출할 페이지 스크린샷이 없습니다.")
        return

    from collections import OrderedDict

    # 합본(AB) 우선, 없으면 A/B로 묶기
    combined = [s for s in shots if s.get("side") == "AB"]
    pairs: OrderedDict = OrderedDict()
    if combined:
        for s in combined:
            label = s.get("pair_label") or f"{s.get('file_a')} = {s.get('file_b')}"
            pairs[label] = s
    else:
        sides_map: OrderedDict = OrderedDict()
        for s in shots:
            label = s.get("pair_label") or f"{s['file_name']} p.{s['page_number']}"
            sides_map.setdefault(label, {"A": None, "B": None, "meta": s})
            sides_map[label][s.get("side", "A")] = s
        for label, sides in sides_map.items():
            pairs[label] = sides

    st.caption(
        f"유사 문장 페이지 쌍 {len(pairs)}개 · "
        "ZIP에는 A|B를 나란히 합친 이미지 1장씩 들어갑니다 "
        "(파일명 예: `파일A_p0003=파일B_p0012.png`)."
    )

    import pandas as pd

    meta_rows = []
    for label, item in pairs.items():
        if isinstance(item, dict) and item.get("side") == "AB":
            meta_rows.append(
                {
                    "비교": label,
                    "파일 A": item.get("file_a", ""),
                    "페이지 A": item.get("page_a", ""),
                    "하이라이트 A": item.get("hits_a", 0),
                    "파일 B": item.get("file_b", ""),
                    "페이지 B": item.get("page_b", ""),
                    "하이라이트 B": item.get("hits_b", 0),
                    "유사문장쌍": item.get("pair_count", ""),
                    "다운로드 파일": item.get("filename", ""),
                }
            )
        else:
            meta = item.get("meta", item)
            a, b = item.get("A") or {}, item.get("B") or {}
            meta_rows.append(
                {
                    "비교": label,
                    "파일 A": meta.get("file_a", ""),
                    "페이지 A": meta.get("page_a", ""),
                    "하이라이트 A": a.get("highlight_hits", 0),
                    "파일 B": meta.get("file_b", ""),
                    "페이지 B": meta.get("page_b", ""),
                    "하이라이트 B": b.get("highlight_hits", 0),
                    "유사문장쌍": meta.get("pair_count", a.get("pair_count", "")),
                    "다운로드 파일": "",
                }
            )
    st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)

    zip_bytes = st.session_state.get("matched_pages_zip")
    if zip_bytes is None:
        zip_bytes = screenshots_to_zip(shots, combined_only=True)
        st.session_state["matched_pages_zip"] = zip_bytes
    st.download_button(
        "matched_pages.zip 다운로드 (나란히 합본)",
        data=zip_bytes,
        file_name="matched_pages.zip",
        mime="application/zip",
        key="dl_matched_pages_tab",
    )

    st.caption(
        "노란색 하이라이트 = 이 페이지 쌍을 만들게 된 유사 문장들의 위치입니다. "
        "A와 B에 같은 글자가 칠해질 필요는 없습니다 — 문장 유사도(의미/표현이 비슷함)로 묶입니다. "
        "아래「묶인 근거」에서 A↔B로 짝지어진 문장을 확인하세요."
    )
    st.markdown("#### 나란히 미리보기")
    for i, (label, item) in enumerate(list(pairs.items())[:30]):
        with st.expander(label, expanded=(i == 0)):
            if isinstance(item, dict) and item.get("side") == "AB":
                st.markdown(
                    f"**A:** {item.get('file_a')} · p.{item.get('page_a')} "
                    f"(하이라이트 {item.get('hits_a', 0)}곳) &nbsp;|&nbsp; "
                    f"**B:** {item.get('file_b')} · p.{item.get('page_b')} "
                    f"(하이라이트 {item.get('hits_b', 0)}곳)"
                )
                st.caption(item.get("filename", ""))
                st.image(item["png_bytes"], use_container_width=True)
                match_pairs = item.get("match_pairs") or []
                with st.expander(
                    f"이 페이지가 묶인 근거 — 유사 문장 쌍 {len(match_pairs) or item.get('pair_count', 0)}개",
                    expanded=True,
                ):
                    if match_pairs:
                        for j, mp in enumerate(match_pairs, 1):
                            sim = mp.get("similarity")
                            sim_s = f"{float(sim) * 100:.1f}%" if sim is not None else "-"
                            st.markdown(
                                f"**{j}.** 유사도 {sim_s}"
                                + (f" · {mp.get('verdict')}" if mp.get("verdict") else "")
                            )
                            c1, c2 = st.columns(2)
                            with c1:
                                st.caption("A")
                                st.write(mp.get("text_a") or "")
                            with c2:
                                st.caption("B")
                                st.write(mp.get("text_b") or "")
                    else:
                        ta, tb = item.get("texts_a") or [], item.get("texts_b") or []
                        st.caption("상세 짝 정보가 없어 하이라이트 대상 문장만 표시합니다.")
                        c1, c2 = st.columns(2)
                        with c1:
                            for t in ta:
                                st.write(t)
                        with c2:
                            for t in tb:
                                st.write(t)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    a = item.get("A")
                    if a:
                        st.markdown(
                            f"**A:** {a['file_name']} · 페이지 {a['page_number']} "
                            f"(하이라이트 {a.get('highlight_hits', 0)}곳)"
                        )
                        st.image(a["png_bytes"], use_container_width=True)
                with c2:
                    b = item.get("B")
                    if b:
                        st.markdown(
                            f"**B:** {b['file_name']} · 페이지 {b['page_number']} "
                            f"(하이라이트 {b.get('highlight_hits', 0)}곳)"
                        )
                        st.image(b["png_bytes"], use_container_width=True)
    if len(pairs) > 30:
        st.info(f"미리보기는 처음 30쌍만 표시합니다. 전체 {len(pairs)}쌍은 ZIP으로 받으세요.")



def render_page_tab(page_pairs):
    """유사 페이지: DataFrame + expander 나란히 비교."""
    if not page_pairs:
        st.info("설정된 기준을 만족하는 유사 페이지가 없습니다.")
        return

    df = page_pairs_to_df(page_pairs)
    st.caption(f"유사 페이지 쌍 {len(df)}개")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### 나란히 비교")
    for i, pair in enumerate(page_pairs):
        title = (
            f"유사도 {pair['similarity'] * 100:.1f}% · {pair['verdict']} · "
            f"{pair['file_a']} p.{pair['page_a']} ↔ {pair['file_b']} p.{pair['page_b']}"
        )
        with st.expander(title, expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**파일 A:** {pair['file_a']}")
                st.markdown(f"**위치:** {pair['location_a']}")
                st.write(pair["text_a"])
            with col2:
                st.markdown(f"**파일 B:** {pair['file_b']}")
                st.markdown(f"**위치:** {pair['location_b']}")
                st.write(pair["text_b"])


def render_sentence_tab(sentence_pairs):
    """결과 표(DataFrame)로 전체 쌍을 먼저 보고, expander로 나란히 비교한다."""
    if not sentence_pairs:
        st.info("설정된 기준을 만족하는 유사 문장이 없습니다.")
        return

    df = sentence_pairs_to_df(sentence_pairs)
    st.caption(f"유사 문장 쌍 {len(df)}개")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### 나란히 비교")
    for i, pair in enumerate(sentence_pairs):
        title = (
            f"유사도 {pair['similarity'] * 100:.1f}% · {pair['verdict']} · "
            f"{pair['file_a']} ↔ {pair['file_b']}"
        )
        with st.expander(title, expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**파일 A:** {pair['file_a']}")
                st.markdown(f"**위치:** {pair['location_a']}")
                st.write(pair["text_a"])
            with col2:
                st.markdown(f"**파일 B:** {pair['file_b']}")
                st.markdown(f"**위치:** {pair['location_b']}")
                st.write(pair["text_b"])


def render_image_tab(image_pairs):
    """결과 표(DataFrame)로 전체 쌍을 먼저 보고, expander로 나란히 미리본다."""
    if not image_pairs:
        st.info("설정된 기준을 만족하는 유사 이미지가 없습니다.")
        return

    df = image_pairs_to_df(image_pairs)
    st.caption(f"유사 이미지 쌍 {len(df)}개")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("#### 나란히 미리보기")
    for i, pair in enumerate(image_pairs):
        title = (
            f"pHash 거리 {pair['phash_distance']} · {pair['verdict']} · "
            f"{pair['file_a']} ↔ {pair['file_b']}"
        )
        with st.expander(title, expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**파일 A:** {pair['file_a']} ({pair['location_a']})")
                st.image(pair["image_bytes_a"], use_container_width=True)
            with col2:
                st.markdown(f"**파일 B:** {pair['file_b']} ({pair['location_b']})")
                st.image(pair["image_bytes_b"], use_container_width=True)


def render_excluded_tab(analysis):
    """자동 제외 목록(등장≥2) + 수동 입력으로 유사 문장 결과 추가 제외."""
    import pandas as pd
    from collections import OrderedDict

    rows = analysis.get("excluded_sentences") or []
    stats = analysis.get("boilerplate_stats") or {}
    multi = [r for r in rows if int(r.get("file_count") or 0) >= 2]

    # --- 수동 제외: form으로 묶어 입력 중 재실행 방지 ---
    st.subheader("수동 제외 (단어·문장 입력)")
    st.caption(
        "한 줄에 하나(또는 쉼표로 구분). 「적용」을 누를 때만 결과에서 제외합니다."
    )
    with st.form("manual_exclude_form", clear_on_submit=False):
        raw_text = st.text_area(
            "제외할 단어 또는 문장",
            height=100,
            placeholder="예:\n주관연구개발기관\n공동연구개발기관",
            key="manual_exclude_text",
        )
        c1, c2, _ = st.columns([1, 1, 2])
        apply_manual = c1.form_submit_button("입력 문구로 결과 제외", type="primary")
        clear_manual = c2.form_submit_button("수동 제외 해제")

    if clear_manual:
        st.session_state.pop("manual_exclude_terms", None)
        st.session_state.pop("manual_exclude_view", None)
        st.session_state["manual_exclude_text"] = ""
        base_pairs = analysis.get("sentence_pairs_base") or analysis.get("sentence_pairs") or []
        # PNG 전체 재생성은 무거우므로 ZIP만 비우고, 다음 다운로드/탭에서 필요 시 생성
        details = collect_matched_page_pair_details(base_pairs)
        pdf_bytes = analysis.get("pdf_bytes_by_name") or {}
        shots = analysis.get("matched_page_pngs") or []
        # base pairs 기준으로 analysis에 이미 있는 pnghots 재사용 시도
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
        terms = parse_manual_exclude_terms(raw_text)
        if not terms:
            st.warning("제외할 단어/문장을 입력하세요.")
        else:
            with st.spinner("입력 문구로 유사 문장 결과·통계·페이지 PNG를 다시 구성하는 중..."):
                view = build_manual_exclude_view(analysis, terms)
            st.session_state["manual_exclude_terms"] = terms
            st.session_state["manual_exclude_view"] = view
            shots = view.get("matched_page_pngs") or []
            st.session_state["matched_pages_zip"] = (
                screenshots_to_zip(shots) if shots else None
            )
            st.rerun()

    terms_applied = st.session_state.get("manual_exclude_terms") or []
    view = st.session_state.get("manual_exclude_view")
    if view is not None and terms_applied:
        base_n = len(analysis.get("sentence_pairs_base") or analysis.get("sentence_pairs") or [])
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

    st.divider()
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

    groups: OrderedDict[str, dict] = OrderedDict()
    for r in multi:
        key = (r.get("normalized_text") or r.get("text") or "").strip()
        if not key:
            continue
        if key not in groups:
            groups[key] = {
                "text": r.get("text") or key,
                "file_count": int(r.get("file_count") or 0),
                "reason": r.get("reason") or "",
                "rows": [],
            }
        groups[key]["rows"].append(r)
        groups[key]["file_count"] = max(
            groups[key]["file_count"], int(r.get("file_count") or 0)
        )

    if not groups:
        st.warning("표시할 유의미 중복 제외 문장이 없습니다.")
    else:
        # 사유 필터는 form 밖 — 옵션 목록만 좁히고, 선택 자체는 form 안에서 재실행 없이
        reason_opts = sorted({g["reason"] for g in groups.values() if g["reason"]})
        reason_filter = st.multiselect(
            "제외 사유 필터",
            options=reason_opts,
            default=reason_opts,
            key="excluded_reason_filter",
        )

        label_to_key: dict[str, str] = {}
        labels: list[str] = []
        for key, g in groups.items():
            if reason_filter and g["reason"] not in reason_filter:
                continue
            label = (
                f"[{g['reason']}] (파일{g['file_count']}) {g['text'][:100]}"
                + ("…" if len(g["text"]) > 100 else "")
            )
            if label in label_to_key:
                label = f"{label} · {key[:20]}"
            label_to_key[label] = key
            labels.append(label)

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
            checked_keys = [label_to_key[lb] for lb in selected_labels if lb in label_to_key]
            if not checked_keys:
                st.warning("다시 포함할 문장을 하나 이상 선택하세요.")
            else:
                settings = st.session_state.get("analysis_settings") or {}
                restore_rows = []
                for key in checked_keys:
                    restore_rows.extend(groups[key]["rows"])
                seen_ids = set()
                uniq_rows = []
                for r in restore_rows:
                    rid = r.get("row_id") or f"{r.get('file_name')}||{r.get('text')}"
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    uniq_rows.append(r)

                base_sentences = list(analysis.get("sentences") or [])
                extra = [_row_to_sentence(r) for r in uniq_rows]
                existing = {
                    f"{s.file_name}||{s.location}||{(s.normalized_text or '').strip()}"
                    for s in base_sentences
                }
                extra = [
                    s
                    for s in extra
                    if f"{s.file_name}||{s.location}||{(s.normalized_text or '').strip()}"
                    not in existing
                ]
                new_sentences = base_sentences + extra
                remaining_excluded = [
                    r
                    for r in rows
                    if (r.get("normalized_text") or r.get("text") or "").strip()
                    not in set(checked_keys)
                ]

                with st.spinner(
                    f"선택 {len(checked_keys)}개 문구({len(extra)}개 문장)를 포함해 "
                    "유사 문장·통계·페이지 PNG를 다시 계산 중..."
                ):
                    updated = recompute_sentence_side(analysis, new_sentences, settings)

                analysis_new = dict(analysis)
                analysis_new.update(updated)
                analysis_new["excluded_sentences"] = remaining_excluded
                st.session_state["analysis"] = analysis_new
                st.session_state["excluded_restore_applied"] = checked_keys
                st.session_state.pop("manual_exclude_view", None)
                st.session_state.pop("manual_exclude_terms", None)
                shots = updated.get("matched_page_pngs") or []
                st.session_state["matched_pages_zip"] = (
                    screenshots_to_zip(shots) if shots else None
                )
                st.success(
                    f"다시 포함 완료: 문장 +{len(extra)} · "
                    f"유사 문장 쌍 {len(updated.get('sentence_pairs') or [])}개"
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
        cols = [c for c in ["파일명", "위치", "제외 사유", "등장 파일 수", "제외 문장"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        st.download_button(
            "excluded_sentences_meaningful.csv 다운로드",
            data=to_csv_bytes(df[cols]),
            file_name="excluded_sentences_meaningful.csv",
            mime="text/csv",
            key="dl_excluded_meaningful",
        )


def main():
    st.title("문서 간 유사 콘텐츠 분석기")
    st.caption(
        "여러 문서를 함께 분석하여 서로 유사한 페이지·문장·이미지를 찾습니다. "
        "분석 결과는 검토 참고용이며 표절 여부를 자동으로 판정하지 않습니다."
    )

    with st.sidebar:
        st.header("설정")
        page_threshold = st.slider(
            "페이지 유사도 기준",
            min_value=0.50,
            max_value=1.00,
            value=DEFAULT_PAGE_THRESHOLD,
            step=0.01,
            help="페이지 전체 텍스트 cosine 유사도 임계값 (초안 기본 0.72)",
        )
        sentence_threshold = st.slider(
            "문장 유사도 기준", min_value=0.50, max_value=1.00,
            value=DEFAULT_SENTENCE_THRESHOLD, step=0.01,
        )
        min_sentence_length = st.number_input(
            "최소 문장 길이", min_value=1, max_value=100, value=DEFAULT_MIN_SENTENCE_LENGTH
        )
        max_sentences = st.number_input(
            "최대 문장 수 (hard-cap)",
            min_value=100,
            max_value=50000,
            value=DEFAULT_MAX_SENTENCES,
            step=100,
            help="대용량 PDF에서 비교에 사용할 문장 상한 (초안 기본 5000)",
        )
        include_same_file = st.checkbox("동일 파일 내부 비교 포함", value=False)
        st.markdown("**양식 문장 필터**")
        exclude_boilerplate_patterns = st.checkbox(
            "양식 패턴·라벨형 문장 제외",
            value=True,
            help="□ 기술 요약, [1차년도…], 기관명 : … 등 서식/라벨 문장을 유사 문장 비교에서 제외합니다.",
        )
        exclude_common_sentences = st.checkbox(
            "여러 파일에 공통인 짧은 양식 문구 제외",
            value=True,
            help=(
                "파일 2개: 둘 다, 3개 이상: N−1개 이상에 동일할 때만. "
                "단, 짧은 양식 라벨만 제외하고 긴 기술 문장은 유사 검토용으로 남깁니다."
            ),
        )
        common_max_length = st.number_input(
            "파일공통 제외 최대 길이",
            min_value=8,
            max_value=80,
            value=20,
            help="이 글자 수 이하이면서 여러 파일에 공통인 짧은 양식 문구만 제외합니다. 기술 본문은 남깁니다.",
            disabled=not exclude_common_sentences,
        )
        enable_image_analysis = st.checkbox("이미지 분석 사용", value=True)
        min_image_width = st.number_input(
            "이미지 최소 가로(px)",
            min_value=1,
            max_value=2000,
            value=DEFAULT_MIN_IMAGE_WIDTH,
            help="초안 기본 180 — 로고/아이콘 제외",
        )
        min_image_height = st.number_input(
            "이미지 최소 세로(px)",
            min_value=1,
            max_value=2000,
            value=DEFAULT_MIN_IMAGE_HEIGHT,
            help="초안 기본 120",
        )
        phash_threshold = st.slider(
            "이미지 pHash 거리 기준", min_value=0, max_value=20,
            value=DEFAULT_PHASH_DISTANCE_THRESHOLD,
        )
        max_results = st.number_input(
            "최대 결과 개수", min_value=10, max_value=5000, value=DEFAULT_MAX_RESULTS
        )

    uploaded_files = st.file_uploader(
        "PDF 파일을 여러 개 업로드하세요", type=["pdf"], accept_multiple_files=True
    )

    analysis = st.session_state.get("analysis")

    # 다운로드 클릭 시 Streamlit이 스크립트를 다시 실행하면서 uploader가 비는 경우가 있음.
    # 그때 early return 하면 download_button이 사라져 ZIP이 내려가지 않음 → 분석 결과가
    # 세션에 있으면 업로드 없이도 결과/다운로드 UI를 유지한다.
    if not uploaded_files and analysis is None:
        st.info("분석할 PDF 파일을 업로드해 주세요.")
        return

    if uploaded_files and len(uploaded_files) < 2:
        st.warning("비교를 위해 2개 이상의 파일을 업로드해 주세요.")

    if uploaded_files and len(uploaded_files) >= 2:
        if st.button("분석 시작", type="primary"):
            settings = {
                "page_threshold": page_threshold,
                "sentence_threshold": sentence_threshold,
                "min_sentence_length": min_sentence_length,
                "max_sentences": int(max_sentences),
                "include_same_file": include_same_file,
                "exclude_boilerplate_patterns": exclude_boilerplate_patterns,
                "exclude_common_sentences": exclude_common_sentences,
                "common_max_length": int(common_max_length),
                "enable_image_analysis": enable_image_analysis,
                "min_image_width": int(min_image_width),
                "min_image_height": int(min_image_height),
                "phash_threshold": phash_threshold,
                "max_results": max_results,
            }
            with st.spinner("문서를 분석하고 있습니다..."):
                analysis = run_analysis(uploaded_files, settings)
            _reset_result_filter_state()
            st.session_state["analysis"] = analysis
            st.session_state["analysis_settings"] = settings
            st.session_state["uploaded_file_names"] = [f.name for f in uploaded_files]
            # ZIP을 미리 만들어 두면 다운로드 클릭 시 재생성 부담·실패를 줄임
            shots = analysis.get("matched_page_pngs") or []
            st.session_state["matched_pages_zip"] = (
                screenshots_to_zip(shots) if shots else None
            )

    analysis = st.session_state.get("analysis")
    if analysis is None:
        return

    if not uploaded_files:
        st.caption("이전 분석 결과를 표시 중입니다. 새 분석은 PDF를 다시 업로드한 뒤 실행하세요.")

    # 수동 제외가 있으면 표시·다운로드에 반영 (원본 analysis는 세션에 유지)
    display = merge_display_analysis(analysis)

    tab_summary, tab_sentences, tab_images, tab_pages, tab_matched, tab_excluded = st.tabs(
        ["분석 요약", "유사 문장", "유사 이미지", "유사 페이지", "페이지 PNG", "제외 문장"]
    )

    with tab_summary:
        n_files = len(display.get("file_names") or st.session_state.get("uploaded_file_names") or [])
        summary_dict = render_summary_tab(display, file_count=n_files)
    with tab_sentences:
        render_sentence_tab(display["sentence_pairs"])
    with tab_images:
        render_image_tab(display["image_pairs"])
    with tab_pages:
        render_page_tab(display.get("page_pairs", []))
    with tab_matched:
        render_matched_pages_tab(display)
    with tab_excluded:
        render_excluded_tab(analysis)

    st.divider()
    st.subheader("결과 다운로드")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.download_button(
            "similar_pages.csv",
            data=to_csv_bytes(page_pairs_to_df(display.get("page_pairs", []))),
            file_name="similar_pages.csv",
            mime="text/csv",
            key="dl_similar_pages",
        )
    with col2:
        st.download_button(
            "similar_sentences.csv",
            data=to_csv_bytes(sentence_pairs_to_df(display["sentence_pairs"])),
            file_name="similar_sentences.csv",
            mime="text/csv",
            key="dl_similar_sentences",
        )
    with col3:
        st.download_button(
            "similar_images.csv",
            data=to_csv_bytes(image_pairs_to_df(display["image_pairs"])),
            file_name="similar_images.csv",
            mime="text/csv",
            key="dl_similar_images",
        )
    with col4:
        shots = display.get("matched_page_pngs") or []
        zip_bytes = st.session_state.get("matched_pages_zip")
        if zip_bytes is None and shots:
            zip_bytes = screenshots_to_zip(shots)
            st.session_state["matched_pages_zip"] = zip_bytes
        if zip_bytes:
            st.download_button(
                "matched_pages.zip (합본)",
                data=zip_bytes,
                file_name="matched_pages.zip",
                mime="application/zip",
                key="dl_matched_pages_footer",
            )
        else:
            st.button("matched_pages.zip", disabled=True, key="dl_matched_pages_disabled")
    with col5:
        excel_bytes = build_excel_report(
            summary_dict,
            display["sentence_pairs"],
            display["image_pairs"],
            display["log_entries"],
            page_pairs=display.get("page_pairs", []),
        )
        st.download_button(
            "similarity_analysis.xlsx",
            data=excel_bytes,
            file_name="similarity_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_excel_report",
        )


if __name__ == "__main__":
    main()
