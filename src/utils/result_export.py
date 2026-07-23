"""분석 결과를 CSV / XLSX로 내보내는 유틸리티."""
from __future__ import annotations

import io
from typing import Optional

import pandas as pd


def sentence_pairs_to_df(pairs: list[dict]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(
            columns=["similarity", "verdict", "file_a", "location_a", "text_a",
                     "file_b", "location_b", "text_b"]
        )
    df = pd.DataFrame(pairs)
    return df[["similarity", "verdict", "file_a", "location_a", "text_a",
               "file_b", "location_b", "text_b"]]


def page_pairs_to_df(pairs: list[dict]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(
            columns=["similarity", "verdict", "file_a", "page_a", "text_a",
                     "file_b", "page_b", "text_b"]
        )
    df = pd.DataFrame(pairs)
    cols = ["similarity", "verdict", "file_a", "page_a", "text_a",
            "file_b", "page_b", "text_b"]
    return df[[c for c in cols if c in df.columns]]


def image_pairs_to_df(pairs: list[dict]) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(
            columns=["phash_distance", "verdict", "file_a", "location_a", "image_id_a",
                     "file_b", "location_b", "image_id_b"]
        )
    # image_bytes는 CSV/요약 표에는 포함하지 않음
    slim = [
        {k: v for k, v in p.items() if k not in ("image_bytes_a", "image_bytes_b")}
        for p in pairs
    ]
    df = pd.DataFrame(slim)
    return df[["phash_distance", "verdict", "file_a", "location_a", "image_id_a",
               "file_b", "location_b", "image_id_b"]]


def processing_log_to_df(log_entries: list[dict]) -> pd.DataFrame:
    if not log_entries:
        return pd.DataFrame(columns=["file_name", "status", "message"])
    return pd.DataFrame(log_entries)[["file_name", "status", "message"]]


def summary_to_df(summary: dict) -> pd.DataFrame:
    rows = [{"항목": k, "값": v} for k, v in summary.items()]
    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # 엑셀 한글 깨짐 방지


def build_excel_report(
    summary: dict,
    sentence_pairs: list[dict],
    image_pairs: list[dict],
    log_entries: list[dict],
    page_pairs: Optional[list[dict]] = None,
) -> bytes:
    """Summary / Similar Pages / Sentences / Images / Processing Log 시트로
    구성된 xlsx 파일을 바이트로 반환합니다."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_to_df(summary).to_excel(writer, sheet_name="Summary", index=False)
        page_pairs_to_df(page_pairs or []).to_excel(
            writer, sheet_name="Similar Pages", index=False
        )
        sentence_pairs_to_df(sentence_pairs).to_excel(
            writer, sheet_name="Similar Sentences", index=False
        )
        image_pairs_to_df(image_pairs).to_excel(
            writer, sheet_name="Similar Images", index=False
        )
        processing_log_to_df(log_entries).to_excel(
            writer, sheet_name="Processing Log", index=False
        )
    buffer.seek(0)
    return buffer.getvalue()
