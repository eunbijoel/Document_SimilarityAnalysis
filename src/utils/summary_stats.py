"""분석 요약용 통계: 문장 겹침 비율, 유사도 구간 분포, 파일×파일 매트릭스."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import pandas as pd

from src.models.schemas import SentenceRecord

_PAGE_NUM = re.compile(r"페이지\s*(\d+)")


def parse_page_number(location: str):
    """'페이지 3' 형태 location에서 페이지 번호를 추출한다."""
    if not location:
        return None
    m = _PAGE_NUM.search(location)
    if not m:
        return None
    return int(m.group(1))


def compute_sentence_overlap_stats(
    sentences: list[SentenceRecord],
    sentence_pairs: list[dict],
) -> dict:
    """전체 문장 중 유사 쌍에 한 번이라도 걸린 문장 수/비율."""
    total = len(sentences)
    # (파일명, location, 정규화/원문)으로 유니크 키 — sentence_id가 있으면 우선
    touched: set[str] = set()
    for pair in sentence_pairs:
        touched.add(f"{pair['file_a']}||{pair['location_a']}||{pair['text_a']}")
        touched.add(f"{pair['file_b']}||{pair['location_b']}||{pair['text_b']}")

    # 추출 문장 기준 매칭 (동일 키 체계)
    sentence_keys = {
        f"{s.file_name}||{s.location}||{s.text}" for s in sentences
    }
    overlapped = len(touched & sentence_keys) if sentence_keys else len(touched)
    # pair 쪽이 truncate 등으로 키가 어긋날 수 있어 touched 크기도 참고
    overlapped = max(overlapped, 0)
    if not sentence_keys and touched:
        overlapped = len(touched)

    ratio = (overlapped / total * 100.0) if total else 0.0

    # 파일별 겹침
    per_file_total: Counter = Counter(s.file_name for s in sentences)
    per_file_hit: Counter = Counter()
    for s in sentences:
        key = f"{s.file_name}||{s.location}||{s.text}"
        if key in touched:
            per_file_hit[s.file_name] += 1

    per_file_rows = []
    for name, tot in per_file_total.items():
        hit = per_file_hit.get(name, 0)
        per_file_rows.append(
            {
                "파일명": name,
                "추출 문장": tot,
                "겹친 문장": hit,
                "겹침 비율(%)": round(hit / tot * 100.0, 2) if tot else 0.0,
            }
        )

    return {
        "total_sentences": total,
        "overlapped_sentences": overlapped,
        "overlap_ratio_pct": round(ratio, 2),
        "pair_count": len(sentence_pairs),
        "per_file_df": pd.DataFrame(per_file_rows),
    }


def similarity_bin_label(score: float) -> str:
    """유사도 점수를 구간 라벨로 변환 (1.0 / 0.9대 / 0.8대 …)."""
    if score >= 0.999:
        return "1.0 (동일)"
    if score >= 0.90:
        return "0.9대"
    if score >= 0.80:
        return "0.8대"
    if score >= 0.70:
        return "0.7대"
    if score >= 0.60:
        return "0.6대"
    return "0.6 미만"


# 그래프용 고정 순서
SIMILARITY_BIN_ORDER = [
    "1.0 (동일)",
    "0.9대",
    "0.8대",
    "0.7대",
    "0.6대",
    "0.6 미만",
]


def compute_similarity_distribution(pairs: list[dict], score_key: str = "similarity") -> pd.DataFrame:
    """유사도 구간별 쌍 개수 DataFrame (차트용)."""
    counts: Counter = Counter()
    for pair in pairs:
        score = float(pair.get(score_key, 0.0))
        counts[similarity_bin_label(score)] += 1

    rows = [
        {"유사도 구간": label, "쌍 개수": counts.get(label, 0)}
        for label in SIMILARITY_BIN_ORDER
        if counts.get(label, 0) > 0 or label in ("1.0 (동일)", "0.9대", "0.8대")
    ]
    # 값이 있는 구간만 + 상위 3구간은 0이어도 표시
    if not any(r["쌍 개수"] for r in rows):
        return pd.DataFrame(columns=["유사도 구간", "쌍 개수"])
    return pd.DataFrame(rows)


def compute_file_pair_matrix(
    sentence_pairs: list[dict],
    file_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    파일×파일 유사 문장 쌍 개수 매트릭스.
    행/열이 파일명, 값이 두 파일 사이 유사 문장 쌍 수.
    """
    names = list(file_names) if file_names else []
    if not names:
        seen = set()
        for p in sentence_pairs:
            seen.add(p["file_a"])
            seen.add(p["file_b"])
        names = sorted(seen)

    if not names:
        return pd.DataFrame()

    matrix = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for p in sentence_pairs:
        a, b = p["file_a"], p["file_b"]
        if a not in matrix.index or b not in matrix.columns:
            # 새 파일명이면 확장
            if a not in matrix.index:
                matrix.loc[a, :] = 0
                matrix[a] = 0
            if b not in matrix.columns:
                matrix.loc[:, b] = 0
                matrix.loc[b, :] = 0
        matrix.loc[a, b] += 1
        if a != b:
            matrix.loc[b, a] += 1
        else:
            # 동일 파일 내부 비교는 대각선에만 +1 (위에서 이미 +1)
            pass

    matrix.index.name = None
    return matrix


def collect_matched_page_keys(sentence_pairs: list[dict]) -> list[tuple[str, int]]:
    """유사 문장 쌍에서 (파일명, 페이지번호) 목록을 중복 없이 수집."""
    keys: set[tuple[str, int]] = set()
    for p in sentence_pairs:
        for file_key, loc_key in (("file_a", "location_a"), ("file_b", "location_b")):
            page = parse_page_number(p.get(loc_key, ""))
            if page is not None:
                keys.add((p[file_key], page))
    return sorted(keys, key=lambda x: (x[0], x[1]))
