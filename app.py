"""문서 간 유사 콘텐츠 분석기

여러 PDF 문서를 업로드하면 동일하거나 유사한 문장/페이지/이미지를 찾아 보여주는
내부 검토용 Streamlit 도구입니다. 표절 여부를 자동으로 판정하지 않습니다.
"""
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

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    total = len(uploaded_files)
    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"처리 중: {uploaded_file.name} ({idx + 1}/{total})")
        try:
            file_bytes = uploaded_file.read()
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

    status_text.text("분석이 완료되었습니다.")
    progress_bar.progress(1.0)

    return {
        "sentences": all_sentences,
        "images": all_images,
        "pages": all_pages,
        "sentence_pairs": sentence_pairs,
        "page_pairs": page_pairs,
        "image_pairs": image_pairs,
        "log_entries": log_entries,
        "sentences_truncated": truncated,
    }


def render_summary_tab(analysis, uploaded_files):
    log_entries = analysis["log_entries"]
    success_count = sum(1 for e in log_entries if e["status"] == "성공")
    fail_count = sum(1 for e in log_entries if e["status"] == "실패")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("파일 수", len(uploaded_files))
    col1.metric("정상 처리", success_count)
    col2.metric("실패", fail_count)
    col2.metric("추출 페이지", len(analysis.get("pages", [])))
    col3.metric("추출 문장", len(analysis["sentences"]))
    col3.metric("유사 페이지 쌍", len(analysis.get("page_pairs", [])))
    col4.metric("유사 문장 쌍", len(analysis["sentence_pairs"]))
    col4.metric("유사 이미지 쌍", len(analysis["image_pairs"]))
    st.metric("추출 이미지", len(analysis["images"]))

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
        import pandas as pd

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
        "추출 문장 수": len(analysis["sentences"]),
        "추출 이미지 수": len(analysis["images"]),
        "유사 페이지 쌍 수": len(analysis.get("page_pairs", [])),
        "유사 문장 쌍 수": len(analysis["sentence_pairs"]),
        "유사 이미지 쌍 수": len(analysis["image_pairs"]),
    }


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

    tab_summary, tab_pages, tab_sentences, tab_images, tab_log = st.tabs(
        ["분석 요약", "유사 페이지", "유사 문장", "유사 이미지", "처리 로그"]
    )

    with tab_summary:
        summary_dict = render_summary_tab(analysis, uploaded_files)
    with tab_pages:
        render_page_tab(analysis.get("page_pairs", []))
    with tab_sentences:
        render_sentence_tab(analysis["sentence_pairs"])
    with tab_images:
        render_image_tab(analysis["image_pairs"])
    with tab_log:
        render_log_tab(analysis["log_entries"])

    st.divider()
    st.subheader("결과 다운로드")
    col1, col2, col3, col4 = st.columns(4)
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
