"""문장·페이지 유사도 분석 모듈.

1) 완전히 동일한 문장은 문자열 비교로 먼저 찾습니다 (임베딩 계산 불필요).
2) 나머지는 다국어 sentence-transformers 임베딩 + NearestNeighbors(top-k)로
   전체 N×N 행렬을 만들지 않고 유사 문장 쌍을 찾습니다.
3) 페이지 전체 텍스트는 초안(compare_page_texts)과 같이 cosine 행렬로 비교합니다.

이 모듈은 Streamlit에 의존하지 않습니다 (테스트 용이성을 위해).
모델 캐싱(st.cache_resource)은 app.py에서 이 모듈의 load_model을 감싸서 처리합니다.
"""
from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.models.schemas import PageRecord, SentenceRecord
from src.utils.config import (
    sentence_verdict,
    DEFAULT_TOP_K,
    EMBEDDING_BATCH_SIZE,
    DEFAULT_PAGE_THRESHOLD,
    PAGE_TEXT_EMBED_MAX_CHARS,
    PAGE_TEXT_PREVIEW_CHARS,
)


def load_model(model_name: str):
    """sentence-transformers 모델을 로드합니다.
    최초 실행 시 인터넷에서 모델을 내려받고, 이후에는 로컬 캐시(~/.cache)를 사용합니다.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def compute_embeddings(model, texts: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # 정규화하면 코사인 유사도 = 내적
        convert_to_numpy=True,
    )
    return embeddings


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def find_exact_duplicate_pairs(
    sentences: list[SentenceRecord], include_same_file: bool
) -> list[dict]:
    """정규화된 텍스트가 완전히 동일한 문장 쌍을 찾습니다."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(sentences):
        groups[record.normalized_text].append(idx)

    pairs = []
    seen: set[tuple[int, int]] = set()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                rec_i, rec_j = sentences[i], sentences[j]
                if not include_same_file and rec_i.file_name == rec_j.file_name:
                    continue
                key = _pair_key(i, j)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(_make_pair_result(rec_i, rec_j, 1.0))
    return pairs, seen


def _make_pair_result(rec_a: SentenceRecord, rec_b: SentenceRecord, score: float) -> dict:
    return {
        "file_a": rec_a.file_name,
        "location_a": rec_a.location,
        "text_a": rec_a.text,
        "file_b": rec_b.file_name,
        "location_b": rec_b.location,
        "text_b": rec_b.text,
        "similarity": round(float(score), 4),
        "verdict": sentence_verdict(score),
    }


def find_similar_sentence_pairs(
    sentences: list[SentenceRecord],
    model,
    threshold: float = 0.80,
    top_k: int = DEFAULT_TOP_K,
    include_same_file: bool = False,
    already_found: Optional[set] = None,
) -> list[dict]:
    """임베딩 + NearestNeighbors를 이용해 유사 문장 쌍을 찾습니다.

    전체 N×N 코사인 행렬을 만들지 않고, 문장별 top-k 이웃만 조회하여
    메모리 사용량을 억제합니다.
    """
    n = len(sentences)
    if n < 2:
        return []

    already_found = already_found or set()
    texts = [s.normalized_text for s in sentences]
    embeddings = compute_embeddings(model, texts)

    k = min(top_k + 1, n)  # +1은 자기 자신 포함
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    results = []
    seen_pairs: set[tuple[int, int]] = set()

    for i in range(n):
        for dist, j in zip(distances[i], indices[i]):
            if j == i:
                continue
            similarity = 1.0 - float(dist)  # cosine distance -> similarity
            if similarity < threshold:
                continue
            key = _pair_key(i, j)
            if key in seen_pairs or key in already_found:
                continue
            rec_i, rec_j = sentences[i], sentences[j]
            if not include_same_file and rec_i.file_name == rec_j.file_name:
                continue
            seen_pairs.add(key)
            results.append(_make_pair_result(rec_i, rec_j, similarity))

    return results


def find_similar_page_pairs(
    pages: list[PageRecord],
    model,
    threshold: float = DEFAULT_PAGE_THRESHOLD,
    include_same_file: bool = False,
    embed_max_chars: int = PAGE_TEXT_EMBED_MAX_CHARS,
    preview_chars: int = PAGE_TEXT_PREVIEW_CHARS,
) -> list[dict]:
    """서로 다른 PDF의 페이지 전체 텍스트 유사도를 계산합니다.

    초안 pdf_similarity_checker.compare_page_texts 와 동일한 접근입니다.
    """
    if len(pages) < 2:
        return []

    texts = [p.text[:embed_max_chars] for p in pages]
    embeddings = compute_embeddings(model, texts)
    # 정규화 임베딩 → 내적 = cosine similarity
    sim = embeddings @ embeddings.T

    results: list[dict] = []
    n = len(pages)
    for i in range(n):
        for j in range(i + 1, n):
            left, right = pages[i], pages[j]
            if not include_same_file and left.file_name == right.file_name:
                continue
            score = float(sim[i, j])
            if score < threshold:
                continue
            results.append(
                {
                    "file_a": left.file_name,
                    "location_a": left.location,
                    "page_a": left.page_number,
                    "text_a": left.text[:preview_chars],
                    "file_b": right.file_name,
                    "location_b": right.location,
                    "page_b": right.page_number,
                    "text_b": right.text[:preview_chars],
                    "similarity": round(score, 4),
                    "verdict": sentence_verdict(score),
                }
            )

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results
