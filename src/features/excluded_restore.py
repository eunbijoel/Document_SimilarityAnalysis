"""자동 제외 문장 복원 — 드롭다운에서 고른 문장을 유사 비교에 다시 포함.

이 모듈은 Streamlit UI와 분리된 순수 로직입니다.
UI는 `src/ui/excluded_sentences_tab.py` 를 참고하세요.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Optional

from src.models.schemas import SentenceRecord

SESSION_RESTORE_APPLIED = "excluded_restore_applied"


def filter_meaningful_excluded(
    excluded_rows: list[dict],
    *,
    min_file_count: int = 2,
) -> list[dict]:
    """등장 파일 수가 min_file_count 이상인 제외 문장만 반환."""
    return [
        r for r in excluded_rows if int(r.get("file_count") or 0) >= min_file_count
    ]


def group_excluded_by_text(rows: list[dict]) -> OrderedDict[str, dict]:
    """
    동일 정규화 문구끼리 묶어 복원 선택용 그룹을 만든다.

    Returns:
        OrderedDict[normalized_text -> {text, file_count, reason, rows}]
    """
    groups: OrderedDict[str, dict] = OrderedDict()
    for r in rows:
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
    return groups


def build_restore_labels(
    groups: OrderedDict[str, dict],
    *,
    reason_filter: Optional[list[str]] = None,
) -> tuple[list[str], dict[str, str]]:
    """
    multiselect용 라벨 목록과 label→key 매핑을 만든다.

    Returns:
        (labels, label_to_key)
    """
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
    return labels, label_to_key


def row_to_sentence(row: dict) -> SentenceRecord:
    """제외 문장 dict → SentenceRecord."""
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


def prepare_restore(
    analysis: dict,
    checked_keys: list[str],
    groups: OrderedDict[str, dict],
) -> tuple[list[SentenceRecord], list[dict], list[SentenceRecord]]:
    """
    선택 키로 복원할 문장·남은 제외 목록을 준비한다.

    Returns:
        (new_sentences, remaining_excluded, extra_added)
    """
    rows = analysis.get("excluded_sentences") or []
    restore_rows: list[dict] = []
    for key in checked_keys:
        restore_rows.extend(groups[key]["rows"])

    seen_ids: set[str] = set()
    uniq_rows: list[dict] = []
    for r in restore_rows:
        rid = r.get("row_id") or f"{r.get('file_name')}||{r.get('text')}"
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        uniq_rows.append(r)

    base_sentences = list(analysis.get("sentences") or [])
    extra = [row_to_sentence(r) for r in uniq_rows]
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
        if (r.get("normalized_text") or r.get("text") or "").strip() not in set(checked_keys)
    ]
    return new_sentences, remaining_excluded, extra


def apply_restore(
    analysis: dict,
    checked_keys: list[str],
    groups: OrderedDict[str, dict],
    *,
    settings: dict,
    recompute_fn: Callable[[dict, list[SentenceRecord], dict], dict],
) -> tuple[dict, list[SentenceRecord]]:
    """
    복원 후 문장 유사도·통계·PNG를 다시 계산한 analysis를 반환.

    Returns:
        (updated_analysis, extra_sentences_added)
    """
    new_sentences, remaining_excluded, extra = prepare_restore(
        analysis, checked_keys, groups
    )
    updated = recompute_fn(analysis, new_sentences, settings)
    analysis_new = dict(analysis)
    analysis_new.update(updated)
    analysis_new["excluded_sentences"] = remaining_excluded
    return analysis_new, extra
