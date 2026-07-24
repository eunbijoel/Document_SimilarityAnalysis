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
    collect_matched_page_keys,
)
from src.utils.page_screenshots import (
    build_matched_page_screenshots,
    screenshots_to_zip,
)

st.set_page_config(page_title="문서 간 유사 콘텐츠 분석기", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model(SENTENCE_MODEL_NAME)


def friendly_error(exc: Exception) -> str:
    """개발자용 traceback 대신 사용자에게 보여줄 간단한 오류 메시지를 만듭니다."""
    return f"{type(exc).__name__}: {exc}"


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

    # --- 문장 hard-cap (대용량 PDF 안전장치, 초안 max_sentences=5000) ---
    max_sentences = settings["max_sentences"]
    truncated = False
    if len(all_sentences) > max_sentences:
        truncated = True
        st.warning(
            f"추출 문장 {len(all_sentences)}개 → 비교는 상위 {max_sentences}개만 사용합니다. "
            "(대용량 PDF 실행시간·메모리 보호용 hard-cap)"
        )
        all_sentences = all_sentences[:max_sentences]
    elif len(all_sentences) > max_sentences * 0.8:
        st.info(f"추출 문장 {len(all_sentences)}개 (상한 {max_sentences}).")

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

    # --- 유사 문장 페이지 PNG 렌더 ---
    status_text.text("유사 문장 페이지 스크린샷 생성 중...")
    page_keys = collect_matched_page_keys(sentence_pairs)
    matched_page_pngs = build_matched_page_screenshots(
        pdf_bytes_by_name, page_keys, dpi=120
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


def _similarity_hist_chart(df, *, height: int = 300):
    """유사도 0.01 단위 히스토그램 (얇은 막대)."""
    import altair as alt

    if df is None or df.empty:
        return

    x_min = float(df["유사도"].min())
    # 막대 너비 = 0.01 스케일 폭에 맞춤
    chart = (
        alt.Chart(df)
        .mark_bar(size=2.5, cornerRadiusEnd=1)
        .encode(
            x=alt.X(
                "유사도:Q",
                scale=alt.Scale(domain=[max(0.0, x_min - 0.01), 1.0], nice=False),
                axis=alt.Axis(
                    title="유사도 (0.01 단위)",
                    format=".2f",
                    tickMinStep=0.01,
                    values=[round(x_min + i * 0.05, 2) for i in range(int((1.0 - x_min) / 0.05) + 1)]
                    + ([1.0] if abs(1.0 - x_min) > 1e-9 else []),
                ),
            ),
            y=alt.Y("쌍 개수:Q", axis=alt.Axis(title="쌍 개수")),
            tooltip=[
                alt.Tooltip("유사도:Q", format=".2f"),
                alt.Tooltip("쌍 개수:Q"),
            ],
        )
        .properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridOpacity=0.25)
    )
    st.altair_chart(chart, use_container_width=True)


def render_summary_tab(analysis, uploaded_files):
    import pandas as pd

    log_entries = analysis["log_entries"]
    success_count = sum(1 for e in log_entries if e["status"] == "성공")
    fail_count = sum(1 for e in log_entries if e["status"] == "실패")
    overlap = analysis.get("overlap_stats") or compute_sentence_overlap_stats(
        analysis["sentences"], analysis["sentence_pairs"]
    )

    st.subheader("기본 지표")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("파일 수", len(uploaded_files))
    col1.metric("정상 처리", success_count)
    col2.metric("실패", fail_count)
    col2.metric("추출 페이지", len(analysis.get("pages", [])))
    col3.metric("추출 문장", overlap["total_sentences"])
    col3.metric("유사 문장 쌍", len(analysis["sentence_pairs"]))
    col4.metric("유사 페이지 쌍", len(analysis.get("page_pairs", [])))
    col4.metric("유사 이미지 쌍", len(analysis["image_pairs"]))

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
    # 세션에 옛(0.1대) 포맷이 남아 있어도 항상 0.01 단위로 재계산
    dist_df = compute_similarity_distribution(analysis["sentence_pairs"])
    if dist_df is not None and not dist_df.empty:
        # 표는 쌍이 있는 구간만 (가독성)
        nonzero = dist_df[dist_df["쌍 개수"] > 0].reset_index(drop=True)
        st.dataframe(nonzero, use_container_width=True, hide_index=True)
        _similarity_hist_chart(dist_df)
        st.caption("X축: 유사도 0.01 단위 · Y축: 해당 유사도 쌍 개수")
    else:
        st.info("유사 문장 쌍이 없어 분포를 그릴 수 없습니다.")

    # --- 3) 파일×파일 매트릭스 ---
    st.subheader("파일 × 파일 유사 문장 쌍 수")
    matrix = analysis.get("file_matrix")
    if matrix is None or (hasattr(matrix, "empty") and matrix.empty):
        names = analysis.get("file_names") or [f.name for f in uploaded_files]
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
        "파일 수": len(uploaded_files),
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
    """유사 문장이 포함된 페이지 PNG 미리보기·ZIP 다운로드."""
    shots = analysis.get("matched_page_pngs") or []
    if not shots:
        st.info("유사 문장 쌍에서 추출할 페이지 스크린샷이 없습니다.")
        return

    st.caption(f"유사 문장이 있는 페이지 {len(shots)}장 (중복 페이지는 1장만 저장)")
    meta = [{"파일": s["file_name"], "페이지": s["page_number"], "파일명": s["filename"]} for s in shots]
    import pandas as pd

    st.dataframe(pd.DataFrame(meta), use_container_width=True, hide_index=True)

    zip_bytes = screenshots_to_zip(shots)
    st.download_button(
        "matched_pages.zip 다운로드",
        data=zip_bytes,
        file_name="matched_pages.zip",
        mime="application/zip",
    )

    st.markdown("#### 미리보기")
    for i, s in enumerate(shots[:30]):  # UI 과부하 방지
        with st.expander(
            f"{s['file_name']} · 페이지 {s['page_number']}",
            expanded=(i == 0),
        ):
            st.image(s["png_bytes"], use_container_width=True)
    if len(shots) > 30:
        st.info(f"미리보기는 처음 30장만 표시합니다. 전체 {len(shots)}장은 ZIP으로 받으세요.")



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


def render_log_tab(log_entries):
    if not log_entries:
        st.info("처리 로그가 없습니다.")
        return
    import pandas as pd

    st.dataframe(pd.DataFrame(log_entries), use_container_width=True)


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

    if not uploaded_files:
        st.info("분석할 PDF 파일을 업로드해 주세요.")
        return

    if len(uploaded_files) < 2:
        st.warning("비교를 위해 2개 이상의 파일을 업로드해 주세요.")
        return

    if st.button("분석 시작", type="primary"):
        settings = {
            "page_threshold": page_threshold,
            "sentence_threshold": sentence_threshold,
            "min_sentence_length": min_sentence_length,
            "max_sentences": int(max_sentences),
            "include_same_file": include_same_file,
            "enable_image_analysis": enable_image_analysis,
            "min_image_width": int(min_image_width),
            "min_image_height": int(min_image_height),
            "phash_threshold": phash_threshold,
            "max_results": max_results,
        }
        with st.spinner("문서를 분석하고 있습니다..."):
            analysis = run_analysis(uploaded_files, settings)
        st.session_state["analysis"] = analysis
        st.session_state["uploaded_file_names"] = [f.name for f in uploaded_files]

    analysis = st.session_state.get("analysis")
    if analysis is None:
        return

    tab_summary, tab_pages, tab_sentences, tab_images, tab_matched, tab_log = st.tabs(
        ["분석 요약", "유사 페이지", "유사 문장", "유사 이미지", "매칭 페이지 PNG", "처리 로그"]
    )

    with tab_summary:
        summary_dict = render_summary_tab(analysis, uploaded_files)
    with tab_pages:
        render_page_tab(analysis.get("page_pairs", []))
    with tab_sentences:
        render_sentence_tab(analysis["sentence_pairs"])
    with tab_images:
        render_image_tab(analysis["image_pairs"])
    with tab_matched:
        render_matched_pages_tab(analysis)
    with tab_log:
        render_log_tab(analysis["log_entries"])

    st.divider()
    st.subheader("결과 다운로드")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.download_button(
            "similar_pages.csv",
            data=to_csv_bytes(page_pairs_to_df(analysis.get("page_pairs", []))),
            file_name="similar_pages.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            "similar_sentences.csv",
            data=to_csv_bytes(sentence_pairs_to_df(analysis["sentence_pairs"])),
            file_name="similar_sentences.csv",
            mime="text/csv",
        )
    with col3:
        st.download_button(
            "similar_images.csv",
            data=to_csv_bytes(image_pairs_to_df(analysis["image_pairs"])),
            file_name="similar_images.csv",
            mime="text/csv",
        )
    with col4:
        shots = analysis.get("matched_page_pngs") or []
        if shots:
            st.download_button(
                "matched_pages.zip",
                data=screenshots_to_zip(shots),
                file_name="matched_pages.zip",
                mime="application/zip",
            )
        else:
            st.button("matched_pages.zip", disabled=True)
    with col5:
        excel_bytes = build_excel_report(
            summary_dict,
            analysis["sentence_pairs"],
            analysis["image_pairs"],
            analysis["log_entries"],
            page_pairs=analysis.get("page_pairs", []),
        )
        st.download_button(
            "similarity_analysis.xlsx",
            data=excel_bytes,
            file_name="similarity_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


if __name__ == "__main__":
    main()
